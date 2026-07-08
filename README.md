# fpgaforge

An **AI-native FPGA implementation engine**. Instead of the traditional report-driven loop
(edit RTL, compile, read reports, edit again), `fpgaforge` exposes a single API:

```python
from fpgaforge import optimize

result = optimize(
    rtl="examples/mac_unpipelined.v",
    top="mac",
    target_fpga="ice40_hx8k",   # roomy package for a wide arithmetic core
    objective="maximize_fmax",
    clock_ns=15,
    iterations=8,
)
print(result.summary())
# e.g. baseline 110.74 MHz -> best 188.22 MHz (+70%) by enabling retiming
```

A real run on this example: the optimizer automatically discovers that
retiming raises Fmax from ~110 MHz to ~188 MHz (and uses fewer LUTs), runs
place-and-route for each candidate, logs every run to the corpus, and returns
the best implementation plus a bitstream.

The optimizer explores implementation knobs, uses a predictive model to rank candidates,
runs the most promising ones through a real open-source FPGA flow, logs every run to a
standardized corpus, and returns the best implementation it found.

## First-pass readiness gate (the headline)

The goal: for **any** input design, know whether it can reach the FPGA on the
first shot — and if not, exactly what is blocking it and how to fix it.
`assess()` fuses every signal the platform produces into one verdict:

```python
from fpgaforge import assess

report = assess("examples/mac_unpipelined.v", top="mac",
                target_fpga="ice40_hx8k", clock_ns=6)
print(report.summary())
```

```
first-pass readiness: AT_RISK  (confidence 85/100)
checks :
  [ok]    synthesis: RTL synthesized to FPGA primitives
  [ok]    place_and_route: placed and routed successfully
  [warn]  timing: tight margin: 188.2 MHz is only +13% over target
  [ok]    resource_headroom: 189/7680 LUTs (2%)
  [ok]    io_fit: 50/~206 package I/O
  [ok]    no_comb_loops / latch_free / clock_domains
  [ok]    functional_bringup: design came up and behaved in the virtual fabric
recommended fixes:
  - Add timing headroom (pipelining/retiming); little margin for board/PVT variation.
```

It runs the (optimizing) implementation flow, a virtual bring-up, and a set of
design-rule checks, then returns a verdict of `READY` / `AT_RISK` / `BLOCKED`
with a 0-100 confidence score and actionable recommendations. The CLI exits
non-zero on `BLOCKED`, so it drops straight into CI or an agent loop:

```bash
fpgaforge assess my_design.v --top my_top --target ice40_up5k --clock-ns 10
```

Checks performed: synthesis, place & route / fit, timing margin, resource
headroom, package I/O fit, combinational-loop and inferred-latch design rules,
clock-domain/CDC heuristic, functional virtual bring-up, and — when the design
fits — a **bitstream-equivalence** check that formally proves (or, failing that,
cycle-accurately verifies) that the flashed bitstream implements your RTL:

```
  [ok]    bitstream_equivalence: flashed bitstream formally equivalent to RTL (all inputs, all time)
```

This is the grounded promise: *maximize first-pass success* by catching most
implementation failures before the first FPGA build — with a READY verdict that
is backed by the actual bits you would flash. (Skip it with `--no-prove`.)

## Reward function for RL (no FPGA in the loop)

The same evidence powers a **reward function** for training a policy that emits
RTL — a drop-in replacement for hooking each rollout up to a real board:

```python
from fpgaforge import score_design

r = score_design("policy_out.v", top="top", clock_ns=10.0)  # optimize=False
r.reward       # dense scalar in [0,1], climbs as the design nears flashable
r.components   # per-stage sub-rewards: timing, resources, equivalence, ...
r.issues       # structured, machine-readable: what to fix, with metric vs target
r.to_dict()    # JSON for your replay buffer
```

```bash
fpgaforge reward policy_out.v --top top --clock-ns 2 --quick --json
```

What makes it a *good* reward (not just the pass/fail gate):

- **Dense & monotone** — partial credit that increases as the design gets closer:
  synthesizing, routing, *approaching* the timing target (continuous
  `fmax/target` shaping), fitting resources/I/O, and proving bitstream
  equivalence. The policy gets gradient long before it fully closes.
- **Scores the emitted RTL, not a tool-tuned variant** — `optimize=False` by
  default, so the reward measures the *policy's* design (normal synthesis still
  runs; that is compilation, not a design change).
- **Structured issues for credit assignment / agent iteration** — each issue
  carries a category, severity, the offending `metric` vs `target`, a suggested
  `fix`, an **evidence tier** (`proven` = tool fact, `modeled` = estimate), and —
  crucially for an agent making the *next edit* — concrete `details` and a
  `location`: the timing **critical path** endpoints + logic-vs-routing
  breakdown (with a depth-aware fix like "8 cell stages → insert a pipeline
  register"), the real synthesizer error with `file:line`, and the equivalence
  **counterexample**. Issues are sorted most-blocking-first.
- **Fatal-fault cap** — designs that fundamentally cannot ship (don't
  synthesize/route, over capacity, or a bitstream that differs from RTL) are
  capped so the policy can't farm partial credit; a *routable* design that only
  misses timing stays fully climbable.
- **Fast/cheap knob** — `quick=True` skips the expensive equivalence step for
  early training; turn it on for the final high-fidelity signal.

Default component weights (override via `weights=`): timing `0.24`, synthesis
`0.12`, routing `0.12`, resources `0.12`, functional `0.10`, io `0.08`, drc
`0.08`, equivalence `0.14`.

The reward also folds in **`modeled`-tier physics** (on by default, `physics=False`
to skip): worst-case **P/V/T** timing headroom and an activity-based **power /
thermal** estimate, surfaced as issues so an agent sees physical risk (thin
margin, junction temp over spec) — not just logic/timing. These are modeled, so
they only mildly scale the reward and never cap a proven signal.

### Caching & parallelism (RL throughput)

The reward is a pure function of (RTL content, top, target, clock, flags), so
repeats — common in RL — are cached and returned instantly. Batches run in
parallel (per-design workdirs never collide):

```python
from fpgaforge import score_design, score_batch, RewardCache

cache = RewardCache()                       # disk-backed, content-addressed
r = score_design("out.v", top="top", cache=cache)   # first: runs the flow
r = score_design("out.v", top="top", cache=cache)   # repeat: instant cache hit

# score a whole rollout batch across worker threads (shared cache)
rewards = score_batch(
    [dict(rtl="a.v", top="a"), dict(rtl="b.v", top="b")],
    max_workers=4, cache=True,
)
```

```bash
fpgaforge reward out.v --top top --quick --cache --json
```

## Structural CDC + timing constraints (SDC)

Clock-domain-crossing bugs pass simulation *and* static timing yet fail on
hardware. `assess()` runs a **structural CDC analysis** on the synthesized
netlist — grouping flops by clock, tracing each capture flop's data cone, and
classifying every crossing as `synchronized` (2-flop), `single_flop`
(metastability risk), or `unsynchronized` (combinational logic on the crossing —
dangerous). Unsynchronized crossings fail the gate and appear as reward issues.

Feed real timing intent with an SDC file (`assess(..., sdc="design.sdc")`):
`create_clock`, `set_false_path`, `set_multicycle_path`, and I/O delays are
parsed so multi-clock designs are understood rather than guessed.

## Timing-accurate emulation (delay-aware)

Functional emulation proves the flashed fabric computes the right values with
*zero* delay. `timing_emulate()` adds the delay dimension: it reads the routed
**SDF** nextpnr emits and runs an independent longest-path solver (reproducing
nextpnr's Fmax to the MHz) to check every capture flop meets setup at the chosen
clock. The verdict fuses both — a design is `TIMING-ACCURATE PASS` only if it is
functionally correct **and** settles at speed; otherwise it reports the exact
failing endpoint and path.

```bash
fpgaforge timing-emulate examples/counter.v --top counter --clock-mhz 100  # PASS
fpgaforge timing-emulate examples/counter.v --top counter --clock-mhz 200  # SETUP VIOLATION @ endpoint
```

**Multi-clock aware.** The SDF solver assigns every flop to a clock **domain**
by tracing the net that drives its clock pin back to its root, then computes a
**per-domain Fmax** — a setup path only constrains a domain when it launches and
captures on the same clock. Register-to-register paths that *cross* domains are
reported separately as clock-domain crossings (their single-clock setup is
meaningless; the structural CDC analysis judges whether they are synchronized),
so a design with two clocks gets an honest Fmax per clock instead of one
falsely-pessimistic number. Single-clock designs are unchanged (one domain,
Fmax matching nextpnr).

## Verifier self-validation (mutation testing)

How do you trust the verifier itself? `mutation_test()` injects faults into the
known-good bitstream and confirms the cycle-accurate comparison catches them.
The **kill rate** is a far more defensible confidence signal than coverage
alone. Two strategies: `netlist` (invert a reconstructed LUT — a valid,
functionally-different fabric that directly tests the comparator) and
`bitstream` (raw tile-bit flips).

```bash
fpgaforge mutation-test examples/counter.v --top counter --mutants 12
# kill rate : 92%  (11 functional kills, 1 masked survivor)
```

## Why

This is the low-risk path toward an AI-native implementation engine described in the project
vision: rather than trying to replace vendor tools wholesale, `fpgaforge` uses existing tools
to collect an enormous corpus of implementation data (timing, placement, routing congestion,
resource usage, outcomes) and builds predictive models and optimization on top of it. Over
time, individual stages can be replaced where our methods consistently outperform.

## Pipeline

```
RTL ─► optimize() ─► [predictive model ranks knob candidates]
                  └─► Backend (Yosys ─► nextpnr ─► icepack)
                       └─► parse reports ─► RunMetrics ─► corpus.jsonl
                                                       └─► train model
                  ◄── best OptimizationResult (knobs, Fmax, bitstream)
```

## Predictive model (Fmax / resources / routability)

The optimizer ranks knob candidates with a `FmaxPredictor` (gradient-boosted
regression over design features + knobs). It stays honest about its own skill:

* **Cross-validated accuracy** — `FmaxPredictor.evaluate(rows)` runs k-fold CV
  and reports MAE / RMSE / R² / within-10% and how much it beats a mean-baseline,
  so you know whether to trust the ranking. `fpgaforge train` prints this.
* **Feature importances** — which design features actually drive Fmax.
* **Three heads** — besides Fmax it predicts **LUT usage** (`predict_luts`) and
  **P(routes)** (`routed_probability`), so the loop can screen out designs likely
  to blow the budget or fail to route *before* spending flow time.
* **Cold start** — until `MIN_SAMPLES` corpus rows exist it falls back to a
  transparent heuristic, so ranking is still sensible on a fresh install.

`fpgaforge bootstrap` (or `fpgaforge.bootstrap_corpus`) sweeps a design across
the whole knob space (and optional targets/clocks) and appends every run to the
corpus — turning an empty corpus into a trainable dataset in one command.

## Virtual FPGA bring-up (pre-hardware)

Before committing a design to a physical board, bring it up on a **virtual
FPGA**: `fpgaforge` synthesizes the RTL down to real FPGA primitives (`SB_LUT4`,
`SB_DFF`, `SB_MAC16`, ...) and runs that *mapped* netlist in a cycle-accurate
fabric (Icarus Verilog + Yosys sim models) inside an auto-generated virtual
board harness (clock, reset, deterministic stimulus, waveform capture).

```python
from fpgaforge import bringup

result = bringup(rtl="examples/counter.v", top="counter", cycles=20)
print(result.summary())
# virtual bring-up: UP
# outputs  : count = 19 (0x13)
# waveform : .runs/bringup/bringup.vcd
```

CLI:

```bash
# Auto harness: detect clk/rst, drive inputs, run N cycles, dump a VCD
fpgaforge bringup examples/mac_pipelined.v --top mac --cycles 12

# Your own testbench with real self-checks (print VFPGA_FAIL to fail the run)
fpgaforge bringup examples/counter.v --top counter --testbench my_tb.v
```

This validates that the *implemented* design actually comes up and behaves,
catching mapping/primitive-inference problems that RTL-only simulation misses.

### Timing sign-off (real per-LUT + routing delays)

Functional bring-up proves the design *behaves*; timing sign-off proves it runs
*at speed*. `signoff()` runs place & route and reads nextpnr's static timing
analysis, which walks the actual per-LUT and per-net delays of the routed
design:

```bash
fpgaforge timing examples/mac_unpipelined.v --top mac --target ice40_hx8k --clock-ns 6
```

```
timing sign-off: VIOLATED
clock  : 110.7 MHz achievable vs 166.7 MHz target
slack  : -3.03 ns
critical path: 9.03 ns = 3.35 ns logic (14 LUT/cell stages) + 5.36 ns routing
  from: a_r_SB_DFFSR_Q_6_DFFLC.O
  to  : y_SB_DFFSR_Q_D_SB_LUT4_O_LC.I3
sdf    : .runs/timing/timing.sdf
```

You get the real critical path broken into **logic (LUT) vs routing** delay,
slack against your target clock, and an emitted **SDF** file for delay-annotated
simulation in an external simulator.

Combine both into one pass with `bringup --timing` — the design must *come up
AND meet timing*:

```bash
fpgaforge bringup examples/counter.v --top counter --timing --clock-ns 10
# virtual bring-up: UP   (count = 19)
# timing   : MET - 143.3 MHz vs 100.0 MHz target (slack +3.02 ns)
# crit path: 6.98 ns (3.75 logic / 2.42 routing)
```

### What virtual bring-up does and does not prove

- Does: mapped-netlist functional behavior, primitive inference, that the design
  compiles/runs and produces expected outputs, with a waveform to inspect.
- Does not: guarantee physical-hardware success. Board-level realities (signal
  integrity, external DDR, peripheral interop, exact timing) still require the
  real board. The honest promise is "eliminate most implementation failures
  before the first FPGA build," not "works first try."
- Timing is signed off from nextpnr STA (real silicon-characterized per-LUT and
  routing delays) and an SDF file is emitted. Full delay-annotated *gate-level*
  waveform simulation of the routed netlist is possible with that SDF but has
  known X-initialization gotchas (carry-chain / control-signal power-up), so
  the authoritative timing verdict comes from STA.

## Bitstream-level fabric emulator (per-device, not per-design)

Everything above simulates a design's *mapped netlist*. The emulator goes one
level lower: it loads the **actual bitstream** — the exact bytes that would be
flashed to the chip — decodes what the silicon fabric is configured to compute,
and runs *that*. This is a per-device model of the iCE40 fabric, built on
Project IceStorm's open bitstream documentation.

```python
from fpgaforge import emulate

emu = emulate("out.bin")   # a real .bin (or .asc) bitstream
print(emu.summary())
```

```
bitstream emulation: out.bin
decoded fabric configuration
device      : iCE40-5k
logic tiles : 660 (5280 cells)
LUTs used   : 9
DFFs used   : 8
carry cells : 7
netlist     : reconstructed -> .runs/emulate/reconstructed.v
```

`emulate()` unpacks the binary, then decodes each logic cell straight from the
config bits — the 16-entry **LUT truth table**, flop enable, and carry — and
reconstructs a simulatable netlist of the configured fabric. The decode is
validated against `icebox_stat` (flop/LUT counts match exactly). You can print
the actual Boolean functions programmed into the LUTs:

```bash
fpgaforge emulate out.bin --show-luts 2
#   tile(15,1) LC1 init=0x6996 dff=True carry=True
#     y = (i0 & ~i1 & ~i2 & ~i3) | ...   # a full-adder sum bit
```

### First-shot verification: does the flashed bitstream match the design?

`verify_bitstream()` is the capstone. It runs RTL all the way to a real `.bin`,
unpacks that **binary** back into the configured fabric, then simulates *both*
the bitstream and the synthesized design under identical stimulus and compares
every cycle:

```python
from fpgaforge import verify_bitstream

v = verify_bitstream("examples/counter.v", top="counter", cycles=32)
print(v.summary())
```

```
bitstream verification: MATCH
design    : add4.add4:...
campaign  : 6 seed(s), 360 concrete cycles compared
coverage  : 100% output-bit toggle (5/5 bits), 31 distinct output vectors
stimulus  : 100% input-bit toggle (8/8 bits) across 2 driven input(s)
bitstream : .runs/verify/out.bin
fabric    : 7 LUTs, 5 DFFs, 5 carry cells
result    : bitstream reproduced the design on every concrete cycle (empirical confidence 81%)
```

**Stimulus quality.** Matching outputs mean little if the campaign never
exercised the inputs, so `verify` now also reports **input-bit toggle coverage**
— the fraction of driven primary-input bits it drove to both 0 and 1 — and folds
it into the confidence score. A design whose inputs barely moved is docked;
purely sequential designs with no free inputs (e.g. a counter) are not penalized.
When a mismatch *is* found, `verify` writes a compact **divergence window**
(`divergence.txt`) showing the stimulus and both sides' outputs for the cycles
around the first diverging cycle, with the offending fields called out — the
minimal counterexample an agent or engineer needs to localize the bug.

```bash
fpgaforge verify examples/counter.v --top counter --cycles 32 --stimulus random
fpgaforge verify examples/counter.v --top counter --stimulus random --seed 1 --seed 99
fpgaforge verify examples/counter.v --top counter --stimulus random --adaptive
```

**Coverage saturation.** With `--adaptive`, the campaign keeps adding seeds
until extra stimulus stops revealing new behavior, then reports whether coverage
*saturated*. This answers the real trust question — "have we tested enough?" —
with evidence rather than a guess. For a design with **no free inputs** (like a
CPU running fixed firmware) coverage is bounded by the design itself, so
saturation means you have observed its *complete reachable* behavior; the RISC-V
SoC below saturates after a handful of seeds at its full 50% gpio usage, and its
confidence rises accordingly.

Because the comparison runs against the bits recovered from the packed binary
(not the pre-pack netlist), a `MATCH` means the image you would flash behaves
like your design. `--stimulus random` runs a **multi-seed campaign**: each seed
drives a different pseudo-random trajectory (both DUTs share the seed, so the
check is fair), and the result reports **measured evidence** — output-bit toggle
coverage, distinct output vectors, and an empirical confidence — instead of a
bare "ran N cycles." A VCD waveform is written for inspection.

**Don't-care–aware comparison.** An uninitialized RTL register or memory reads
as `x` (a don't-care); the real fabric powers up to a concrete value, which is a
*legal refinement*. `verify` therefore only fails when *both* sides are concrete
and differ, and it counts (not conflates) the don't-care cycles it skipped. This
is why a BRAM design — whose RTL memory starts unknown while the bitstream BRAM
inits to `0` — now verifies correctly instead of reporting a spurious mismatch.

Designs whose I/O exceeds the package pin count report a clear error (e.g.
`design needs 50 IO pins but package only exposes 39`) — a real physical
constraint, surfaced before hardware.

### Formal proof: bitstream ≡ RTL for *all* inputs

`verify` compares one stimulus sequence. `prove` goes further and *formally*
proves the flashed bitstream computes the same thing as your RTL for **every
possible input** — using a SAT-based miter with temporal induction (an all-time
proof), falling back to bounded model checking:

```python
from fpgaforge import prove

p = prove("examples/counter.v", top="counter")
print(p.summary())
```

```
formal equivalence: PROVEN EQUIVALENT
design    : counter.counter:...
method    : induction (sat engine)
bitstream : .runs/prove/out.bin
fabric    : 9 LUTs, 8 DFFs, 7 carry cells
result    : the flashed bitstream is formally equivalent to the design for ALL inputs and ALL time (unbounded induction)
```

```bash
fpgaforge prove examples/counter.v --top counter          # unbounded (induction)
fpgaforge prove examples/counter.v --top counter --bounded --depth 32
fpgaforge prove examples/ram_sync.v --top ram_sync --strategy smt --depth 4
```

This is the strongest software-level guarantee available: it proves the entire
toolchain (synthesis → place & route → bitstream pack) preserved your design's
semantics, so the exact `.bin` you flash is mathematically equivalent to your
RTL — not just "equal on the tests we ran." If the designs ever differ, `prove`
returns `NOT EQUIVALENT` with a counterexample.

**Two proof engines (memory scaling).** The `sat` engine bit-blasts the miter —
fast for logic, but a memory becomes per-bit state that explodes the SAT
problem. The `smt` engine instead keeps memories as **native SMT arrays**
(`yosys-smtbmc` + z3), so a design with on-chip RAM stays tractable: it proves
equivalence over `--depth` cycles from a zero-initialized reset state for all
inputs. `--strategy auto` (the default) picks `smt` for memory-bearing designs
when the SMT tools are installed, and `sat` otherwise. Because a physical
`SB_RAM40_4K` is reshaped relative to the RTL array, the SMT proof depth trades
off against solve time — a shallow bounded proof is still a real all-inputs
guarantee the SAT engine cannot produce for memory at all.

**The trust ladder.** Formal proof strength depends on the design. Pure logic
proves for all time by induction. On-chip memory is harder: the `sat` engine
would bit-blast a BRAM into per-bit state that explodes for anything CPU-sized
(a single 256×16 RAM is already millions of clauses), so the `smt` engine keeps
the memory as an array and proves a *bounded* number of cycles instead — a real
all-inputs guarantee, but bounded because a physical `SB_RAM40_4K` is reshaped
relative to the RTL `$mem`. So the platform reports the *strongest evidence it
can actually establish*:

| Rung | Guarantee | Reached by |
|------|-----------|-----------|
| **proved (all time)** | bitstream ≡ RTL for every input, forever | `prove` induction (pure-logic designs) |
| **proved (bounded)** | bitstream ≡ RTL for every input, N cycles from reset | `prove` BMC (`sat` for logic, `smt`-arrays for memory designs) |
| **verified (measured)** | bitstream matched RTL on every concrete cycle across a multi-seed campaign, with reported coverage/confidence and coverage-saturation | `verify --stimulus random --adaptive` (deep memory/CPU designs) |

`assess` climbs this ladder automatically: it tries `prove` first and, when a
memory-heavy design can't be proven to a useful depth in time, falls back to the
measured `verify` campaign — so the `READY` verdict is always backed by the best
available bitstream evidence, with the exact rung named in the report.

### Netlist-level proof: RTL ≡ vendor post-implementation netlist

On iCE40 the bitstream is an open format, so `prove` reconstructs the fabric and
proves RTL ≡ *the actual flashed bits* (AMD 7-series and Gowin reach the same
tier via Project X-Ray / Apicula when those tools are installed). On locked
silicon (AMD UltraScale+, all Intel) the bitstream is encrypted/undocumented
and cannot be reconstructed — but the
vendor tool emits a **gate-level post-implementation netlist** (Vivado
`write_verilog -mode funcsim`, Quartus `.vo`) built from documented library
primitives (`LUT6`, `FDRE`, `CARRY4`, `RAMB36E1`, `DSP48E1`, …). yosys ships
behavioral models of exactly those primitives, so `prove` can run the *same*
miter — RTL vs the routed netlist — one rung below the bits:

```python
from fpgaforge import prove

# RTL vs the netlist your AMD/Intel flow already produced.
p = prove("cpu.v", top="cpu", target_fpga="xc7a35t",
          netlist="impl/cpu_funcsim.v")
print(p.summary())
# result: the post-implementation netlist is formally equivalent to the design
#         for ALL inputs and ALL time (unbounded induction)
```

```bash
fpgaforge prove cpu.v --top cpu --target xc7a35t --netlist impl/cpu_funcsim.v
```

This proves synthesis + place & route preserved your semantics through the
netlist — everything except the final, undocumented bit-packing step. `assess`
wires it in automatically: for a vendor-locked device whose backend emitted a
Verilog netlist, the readiness gate runs the netlist-level proof and reports a
`netlist_equivalence` check; otherwise it names netlist equivalence as the
strongest achievable tier. The sim library per family lives in the device
registry (`DeviceInfo.sim_lib`), so the same flow lifts ECP5, 7-series/UltraScale+
and Cyclone/MAX10 designs to the netlist tier with no vendor tools at prove time.

### Proven on a real RISC-V CPU (picorv32)

`fpgaforge` handles real, third-party designs — not just toy examples. Pull the
[picorv32](https://github.com/YosysHQ/picorv32) core and point the readiness
gate at it:

```bash
fpgaforge assess examples/third_party/picorv32.v --top picorv32 --target ice40_hx8k
# first-pass readiness: BLOCKED  (confidence 0/100)
#   [ok]    resource_headroom: 1815/7680 LUTs (24%)
#   [FAIL]  io_fit: needs 409 I/O but package has ~206
#   tool errors: nextpnr-ice40: Unable to find a placement location for 'pcpi_insn[17]$sb_io'
```

That is the honest, actionable verdict: a raw CPU core's 409-bit memory
interface cannot pin out on any iCE40 package — caught before hardware. Wrap it
in a small SoC (`examples/third_party/picosoc_lite.v`: picorv32 + on-chip RAM +
a memory-mapped GPIO, preloaded with a tiny program that counts on the GPIO),
and the whole engine lights up on a genuine RISC-V processor:

```bash
SRC="examples/third_party/picosoc_lite.v examples/third_party/picorv32.v"

# Runs the program on the virtual fabric
fpgaforge bringup $SRC --top picosoc_lite --cycles 400
# virtual bring-up: UP   (gpio = 25)

# A real CPU closes timing on the fabric
fpgaforge timing $SRC --top picosoc_lite --target ice40_up5k --clock-ns 40
# timing sign-off: MET   28.9 MHz vs 25.0 MHz  (critical path through cpu.reg_op2 -> cpu.mem_do_rinst)

# The flashed bitstream runs the program cycle-identically to the design
fpgaforge verify $SRC --top picosoc_lite --target ice40_up5k --cycles 200 --clock-mhz 12
# bitstream verification: MATCH   (1557 LUTs, 548 DFFs, 240 carry cells)
```

The program lives in the bitstream's BRAM init and is recovered directly from
the packed binary — so a `MATCH` means the exact image you would flash executes
the RISC-V program identically to the source design.

## Virtual board: run the bitstream against real peripherals

A bitstream on its own is just logic. A *real* FPGA is soldered to a board: an
oscillator drives its clock, a reset controller wakes it, and its pins talk to
LEDs, a UART, buttons, switches, and GPIO. `emulate_board` models that board —
it builds the real bitstream, reconstructs the fabric, then **wires that fabric
to behavioral peripheral models and runs it**, so you can read the UART text and
watch the LEDs the flashed image actually drives.

```python
from fpgaforge import emulate_board, BoardConfig

r = emulate_board("examples/blinky_uart.v", top="blinky_uart",
                  board=BoardConfig(clock_mhz=12, baud=1_000_000, duration_us=60))
print(r.summary())
```

```
virtual board: RAN
design : blinky_uart.blinky_uart:...
bitstream: .runs/board/out.bin
peripherals: led, uart_tx
uart[tx]: "Hi\nHi" (5 bytes)
led[led]: 11 toggle(s), final=1
waveform: .runs/board/board.vcd
```

```bash
fpgaforge board examples/blinky_uart.v --top blinky_uart --clock-mhz 12 --baud 1000000
fpgaforge board my_uart_echo.v --top top --uart-send "ping"   # transmit into the DUT's rx
```

Pins are auto-mapped to peripherals by name (`led*` → LED, `tx`/`rx` → UART,
`btn`/`key` → button, `sw`/`dip` → switch, everything else → GPIO), overridable
via `BoardConfig.pin_roles`. The peripheral models are real behavioral Verilog:

- **Clock oscillator** at `clock_mhz`, **reset controller** (auto active-high/low).
- **UART** — a time-based receiver decodes the bytes the DUT transmits into ASCII
  (start/8-data/stop at `baud`), and a transmitter injects bytes into the DUT's
  `rx` (`--uart-send` / `uart_rx_bytes`).
- **LED / GPIO monitors** log every transition with a timestamp (blink counts,
  final levels).
- **Button / switch drivers** apply static levels or timed press schedules.

Because it drives the reconstructed *bitstream* (not the RTL), this is the "FPGA
in a socket" experience — the same real RISC-V SoC boots on the virtual board and
its firmware counts up on the GPIO:

```python
r = emulate_board(["examples/third_party/picosoc_lite.v",
                   "examples/third_party/picorv32.v"],
                  top="picosoc_lite", target_fpga="ice40_hx8k",
                  board=BoardConfig(clock_mhz=12, duration_us=40))
# gpio[gpio]: final=30   (0,1,2,3,... straight from the flashed image's firmware)
```

A VCD of the whole board (DUT pins + peripherals) is written for inspection.
This is emulation of the reachable, on-chip behavior. The genuinely *physical*
effects — PVT-dependent Fmax, signal-integrity overshoot/ringing, external
interface setup/hold — are modeled by the physical sign-off layer below.

## Physical sign-off (beyond the netlist)

Meeting logical and STA timing is not the same as working on a board. The
`physics` layer models, to first/second order, the effects a netlist cannot see,
turning them into quantified margins and risk flags:

```bash
fpgaforge physics examples/blinky_uart.v --top blinky_uart --clock-ns 40 \
    --board-file examples/board_spec.json
```

```
physical sign-off: AT_RISK

PVT timing sign-off
STA Fmax   : 55.1 MHz (assumed slow corner)
guaranteed : 55.1 MHz across P/V/T (MEETS worst case, +120% margin)
  slow_hot   P=slow V=1.14 T=+85C ->  55.1 MHz  (x1.000)
  typical    P=typ  V=1.20 T=+25C ->  85.2 MHz  (x0.646)
  fast_cold  P=fast V=1.26 T=+0C  -> 130.0 MHz  (x0.424)

signal integrity [uart_tx]: AT RISK
  overshoot      : 33% of Vdd (peak 4.40 V)
  settling       : 3.27 ns -> safe toggle <= 153 MHz
  recommendation : add source series termination to match Z0
  [risk] overshoot 33% exceeds 20% limit -> reliability/latch-up risk

interface [uart]: CLOSES
  eye opening   : 993.000 ns (UI - 7.000 ns uncertainty)
  total margin  : +953.000 ns
```

Four physically-grounded models, all available as a Python API too
(`from fpgaforge import derate_fmax, analyze_net, analyze_crosstalk, analyze_interface, physical_signoff`):

- **PVT derating** (`pvt.py`) — derates the STA Fmax across process/voltage/
  temperature via an alpha-power delay law, reporting the *guaranteed* slow-corner
  Fmax you can promise. Because `nextpnr-ice40`'s STA is already the slow corner,
  the guaranteed number equals it and the other corners show the upside; set
  `--sta-corner typ` for flows that report a typical number.
- **Signal integrity** (`signal_integrity.py`) — per-net rise time, transmission-
  line reflections/overshoot (`(Z0−Zout)/(Z0+Zout)`), package LC ringing, and
  simultaneous-switching noise (`N·L·dI/dt`), with termination advice and a safe
  toggle-rate ceiling from the settling time. **If `ngspice` is on PATH it runs a
  real transient circuit simulation** (Thevenin driver → T-line → load), dumps the
  waveform, and folds the measured overshoot and settling back into the verdict
  (`ngspice-verified` in the report).
- **Crosstalk** (`crosstalk.py`) — coupled-line near-end (NEXT) and far-end (FEXT)
  noise from an aggressor onto a victim: `NEXT = ¼(k_c+k_l)·Vdd`, saturating for
  long coupled runs, `FEXT = ½|k_l−k_c|·(Td/tr)·Vdd`, compared to the victim's
  noise margin.
- **Interface budgets** (`interfaces.py`) — source-synchronous / DDR eye budget:
  `eye = UI − (tco_spread + skew + jitter + dcd)` vs `tSU + tH`, with setup/hold
  margins.
- **Power delivery / PDN** (`pdn.py`) — the decoupling network (VRM + bulk + mid +
  HF ceramics, each a series R/ESL/C) presents an impedance `Z(f)` to the die.
  It sweeps `|Z(f)|` against the target impedance
  `Z_target = Vdd·ripple / I_transient` and flags **anti-resonance peaks** where
  the rail would droop, reporting each bank's self-resonant frequency. The VRM
  sets the low-frequency floor.
- **Power & thermal, closing the PVT loop** (`power.py`) — activity-based dynamic
  power (`Σ a·C·V²·f` over LUTs/FFs/BRAM/DSP + I/O) plus temperature-dependent
  leakage give the total dissipation; a `Tj = Tamb + P·θ_JA` fixed point (leakage
  depends on `Tj`, which depends on leakage — thermal runaway is detected) yields
  the real junction temperature. That self-heated `Tj` is fed **back into the PVT
  hot corner**, so the guaranteed Fmax is thermally self-consistent instead of
  assuming a fixed 85 °C. Supply a `"power"` block in the board spec (resources,
  I/O count, θ_JA, ambient).
- **Measured switching activity** (`vcd.py`) — the activity factor `a` is the
  biggest lever in dynamic power, and assuming a global constant is a guess. The
  virtual bring-up / verify / board runs already dump a full VCD of the mapped
  fabric, so `measure_activity(vcd)` counts per-bit transitions of every net and
  divides by the clock-cycle count to get the design's *real* mean toggle rate.
  Pass `--vcd <waveform>` (or `--measure-activity` to auto-run a bring-up) to
  `fpgaforge physics` and power is grounded in how the design actually behaves
  under its own stimulus instead of a datasheet default.

**Geometry-driven field solver (Z0, velocity, and coupling from the stackup).**
Rather than typing in a trace impedance and coupling ratios, give the field
solver (`fieldsolver.py`) the PCB *stackup geometry* and it derives them from
accepted quasi-static closed forms:

- single-line **Z0 + propagation velocity** — microstrip (Hammerstad-Jensen) and
  symmetric stripline (Cohn / IPC-2141);
- coupled-line **even/odd mode impedances** (Garg-Bahl static capacitances) →
  the inductive/capacitive coupling ratios `k_l`, `k_c` the crosstalk model
  consumes, so NEXT/FEXT come from edge spacing, not a guess.

```json
{ "name": "uart_tx",
  "geometry": { "kind": "microstrip", "trace_w_mm": 0.30, "height_mm": 0.17, "er": 4.3 },
  "load_pf": 8, "trace_len_mm": 90 }
```

A typical `w=0.30 mm / h=0.17 mm` FR-4 microstrip solves to ~53 Ω at 0.56 c, and
coupling falls off monotonically as edge spacing grows — exactly as measured on
real boards. Use directly via `from fpgaforge import microstrip_line,
coupled_microstrip, net_from_geometry, crosstalk_from_geometry`.

**IBIS driver models (measured, not guessed).** Instead of a generic driver
impedance and rise time, point a net at a vendor **IBIS** model and the SI
analysis uses the datasheet's measured pull-up/pull-down I-V curves, edge ramp,
and package parasitics:

```json
{ "name": "uart_tx",
  "ibis": { "file": "ice40_lvcmos33.ibs", "model": "LVCMOS33_8mA" },
  "load_pf": 8, "trace_z0_ohm": 50, "trace_len_mm": 90 }
```

This is the biggest fidelity jump short of transistor SPICE — e.g. an 8 mA
buffer's true ~70 Ω output impedance does *not* overshoot a 50 Ω line, where a
hand-guessed 25 Ω would have falsely flagged it. Load via
`from fpgaforge import load_ibis, net_from_ibis`.

**The honest boundary.** These are models, not measurements. Their accuracy is
bounded by the board/package parameters you supply (`examples/board_spec.json`
and `examples/ice40_lvcmos33.ibs` show the schema), and no software fully
captures manufacturing spread, 3D-EM field coupling, or environmental effects.
This layer *shrinks and quantifies* physical risk and tells you exactly what to
fix — it does not replace a hardware bring-up. The quasi-static field solver is
accurate to a few percent for typical geometries but a full 3-D EM extraction on
the real layout remains the gold standard for tight designs, and the PDN model
is a lumped network — good for choosing decoupling and catching anti-resonance,
not a plane-cavity resonance solve.

## Pin constraints: the board-level gate

A wrong or missing pin assignment is the classic first-shot killer: the design
is functionally perfect, the tools auto-place I/O on whatever pins route best,
and the board is wired to different ones. `fpgaforge` makes the pin map a
first-class, *checked* input:

```bash
fpgaforge assess examples/counter.v --top counter \
    --pins examples/counter_up5k.pcf --board examples/board_spec.json
```

```
[ok]  pin_constraints: pin map validated: 10/10 port bits pinned,
      validated against the package and board spec
```

The checker parses every supported dialect — `.pcf` (iCE40), `.lpf` (ECP5),
`.xdc` (AMD/Vivado), `.qsf` (Intel/Quartus) — and validates against three
levels of ground truth:

1. **The design's real ports** (from the synthesized netlist): every bit of
   every top-level port must be pinned; no constraints for ports that don't
   exist; no double-booked pins. A partial bus (`count[0]` pinned, bits 1–7
   auto-placed) is a hard failure.
2. **The package**: pins must exist on the device package (checked against the
   IceStorm chipdb on iCE40) and fit the I/O budget. On iCE40 the `.pcf` is
   also fed to the *real* place & route, so the proven bitstream is constrained
   to the board's pins — not auto-placed.
3. **The board spec** (`--board`): the clock port must be pinned to an actual
   board clock source (and a too-slow source warns that a PLL is needed), and
   each I/O standard's voltage must be backed by a real board rail
   (`LVCMOS33` with no 3.3 V rail is a failure).

`--require-pins` makes a missing pin map a hard `BLOCKED` — recommended for any
gate that feeds real hardware. Programmatic API: `load_pins()` / `check_pins()`.

## On-FPGA self-test: verifying the silicon itself

Silicon defects, marginal rails, and environment (temperature, clock quality)
are *physically outside any simulator's reach*. So the platform generates the
artifact that can check them — a built-in self-test that runs on the real chip:

```bash
fpgaforge selftest examples/counter.v --top counter --cycles 4096 --build
```

```
on-FPGA self-test harness generated
golden   : 0x81805224d6272233
coverage : 4096 pseudo-random cycles after 8-cycle reset
validated: yes -- harness simulation asserts test_pass
harness bitstream: .runs/selftest/build/out.bin
```

The harness wraps your design with a maximal-length **LFSR** driving every
input and a **MISR** compressing every output of every cycle into a 64-bit
signature, compared against a **golden signature** computed cycle-accurately in
simulation and baked into the bitstream. On the bench: `test_done` high +
`test_pass` high means the physical silicon reproduced the simulation
bit-for-bit; `test_pass` low means a silicon/board-level fault. The generator
validates its own harness in simulation, is proven to catch modeled defects
(a stuck output bit drops `test_pass`), and *refuses to generate* when the
design has non-reset state that would make the predicted signature unsound.

This closes the loop end-to-end: formal proof guarantees the bits implement
your RTL, and the self-test verifies the chip executes those bits correctly in
its real electrical environment.

## Backends and multi-vendor support

Every target the platform knows about lives in one **device registry**
(`fpgaforge/devices.py`) as a `DeviceInfo` — vendor, family, backend, capacities
(LUT/FF/BRAM/DSP/IO), and, crucially, *which capability tiers it can reach*.
`backend_for_target()` and every capacity table are derived from it, so adding a
part is a one-line registry entry.

- **`Ice40Backend`** — open-source Lattice iCE40 flow (`yosys synth_ice40` →
  `nextpnr-ice40` → `icepack`, Project IceStorm).
- **`Ecp5Backend`** — open-source Lattice ECP5 flow (`synth_ecp5` →
  `nextpnr-ecp5` → `ecppack`, Project Trellis). Targets `ecp5_12k/25k/45k/85k`.
- **`VivadoBackend`** — AMD/Xilinx flow. Drives Vivado in `-mode batch`
  (`synth_design` → `opt/place/route` → `report_{utilization,timing,power}` →
  `write_verilog`/`write_sdf`/`write_bitstream`) and parses the reports into the
  same `RunMetrics`. Targets 7-series/Zynq/UltraScale+ (`xc7a35t`, `xc7z020`,
  `xczu3eg`, …).
- **`QuartusBackend`** — Intel/Altera flow (`quartus_map` → `quartus_fit` →
  `quartus_sta` → `quartus_pow`), parsing the `.rpt` files. Targets Cyclone/MAX 10
  (`cyclonev_5csema5`, `max10_10m50`, …).
- **`MockBackend`** — deterministic, tool-free backend so the API, model, and
  tests run fully offline. `backend_for_target()` falls back to it when a
  vendor's tools aren't installed.

### Support is *tiered*, honestly

Bit-level bring-up — decoding the *flashed bitstream* back to a fabric and
proving it ≡ RTL — is only possible where an **open bitstream database** exists.
Vendor-locked silicon (AMD UltraScale+, all Intel) ships an
undocumented/encrypted bitstream, so that guarantee is *physically* impossible
there. Each device therefore advertises the strongest tier it can reach, via a
pluggable `FabricReconstructor`:

- `IceStormReconstructor` — iCE40 via `icebox_vlog` (always available with the
  open flow).
- `XRayReconstructor` — AMD 7-series via **Project X-Ray** (`bit2fasm` +
  `fasm2bels`, with `PRJXRAY_DB_DIR` pointing at the database checkout). The
  7-series bitstream is community-documented, so these parts are
  bit-reconstructable when the tools are installed.
- `ApiculaReconstructor` — Gowin LittleBee via **Project Apicula**
  (`gowin_unpack`, `pip install apycula`), one-step bitstream → Verilog decode.
- `NoReconstruction` — truly locked silicon (UltraScale+, Intel).

| Equivalence tier | What's proven | Devices |
|---|---|---|
| **bitstream** | RTL ≡ the actual bitstream you flash | iCE40 (IceStorm), AMD 7-series (X-Ray, tool-gated), Gowin (Apicula, tool-gated) |
| **netlist** | RTL ≡ vendor post-implementation netlist (formal miter, + vendor timing/power sign-off, measured `verify`) | ECP5, AMD (Vivado), Intel (Quartus), Gowin |
| **none** | implementation + timing only | (mock) |

The tier is **availability-aware**: `DeviceInfo.equivalence_tier` states what
the device's bitstream *format* permits, and `achievable_tier()` narrows it to
what the installed toolchain can do right now — so an xc7 part honestly reports
the netlist tier until prjxray is installed, with a message naming exactly what
to install to climb to the bit level.

The **netlist** tier is a real formal proof, not just measured verification:
`prove(..., netlist=...)` miters the RTL against the routed gate-level netlist
using yosys' bundled vendor sim libraries (`DeviceInfo.sim_lib`). `assess` runs
it automatically when a vendor backend emits a Verilog netlist and reports a
`netlist_equivalence` check. `assess` names the tier per device (e.g. `device :
ice40_up5k [lattice] - equivalence tier: bit-level`), and `verify`/`prove`
degrade gracefully with a clear message on vendor parts instead of pretending to
do the impossible.

## Install

Python package:

```bash
python3 -m pip install -e ".[dev]"
```

System tools for the real iCE40 flow (macOS / Homebrew):

```bash
brew install yosys nextpnr-ice40 icestorm
```

If the system tools are not present, everything still works against `MockBackend`.

## CLI

```bash
# Single flow run, appends metrics to the corpus
fpgaforge run examples/counter.v --top counter --target ice40_up5k

# Optimize a design toward maximum Fmax
fpgaforge optimize examples/mac_unpipelined.v --top mac \
    --target ice40_hx8k --clock-ns 15 --iterations 8

# Grow the training corpus by sweeping a design across the knob space
# (repeat --target / --clock-ns to fan out; --mock for an offline dry run)
fpgaforge bootstrap examples/mac_unpipelined.v --top mac \
    --target ice40_up5k --clock-ns 10 --clock-ns 6 --seed 1 --seed 2

# Train the Fmax predictor and report cross-validated accuracy
fpgaforge train

# Load a real bitstream and decode/run the configured fabric
fpgaforge emulate out.bin --show-luts 4

# Prove the flashed bitstream matches the design, cycle by cycle
fpgaforge verify examples/counter.v --top counter --cycles 32

# Formally prove the bitstream equals the RTL for ALL inputs
fpgaforge prove examples/counter.v --top counter

# Run the flashed bitstream on a virtual board of peripherals (UART, LEDs, GPIO)
fpgaforge board examples/blinky_uart.v --top blinky_uart --clock-mhz 12 --baud 1000000

# Physical sign-off: PVT-derated Fmax, signal integrity, interface budgets
fpgaforge physics examples/blinky_uart.v --top blinky_uart --clock-ns 40 \
    --board-file examples/board_spec.json

# ...with switching activity measured from the design's own waveform
fpgaforge physics examples/blinky_uart.v --top blinky_uart --clock-ns 40 \
    --board-file examples/board_spec.json --measure-activity

# Score a design as an RL reward: dense scalar + structured issues (JSON for the loop)
fpgaforge reward examples/counter.v --top counter --clock-ns 2 --quick --json --cache

# Timing-accurate emulation: functional match + real-delay setup at a clock
fpgaforge timing-emulate examples/counter.v --top counter --clock-mhz 100

# Validate the verifier itself: inject bitstream faults, report the kill rate
fpgaforge mutation-test examples/counter.v --top counter --mutants 12

# Board-level gate: validate the pin map against ports, package, and board
fpgaforge assess examples/counter.v --top counter \
    --pins examples/counter_up5k.pcf --board examples/board_spec.json --require-pins

# Generate + build an on-FPGA self-test (BIST) that verifies the silicon itself
fpgaforge selftest examples/counter.v --top counter --cycles 4096 --build
```

Add `--mock` to any command to force the offline `MockBackend`.

### Device notes

- `ice40_up5k` (sg48) is a small 48-pin package: great for compact designs like
  `counter`, but wide I/O cores (e.g. `mac`, 50 pins) need a roomier package
  such as `ice40_hx8k` (ct256) or a pin-constraint file.
- Only the UltraPlus family (`ice40_up5k`) has hardened DSP (`SB_MAC16`) blocks;
  the DSP knob is automatically ignored on other families.
- A design with no register-to-register paths reports "No Fmax available" from
  nextpnr; the example cores register both inputs and outputs so timing is
  meaningful.

## Roadmap

`fpgaforge` is intentionally a thin but complete vertical slice. Each seam grows independently:

- **More targets**: iCE40 + ECP5 + Gowin (open source) and AMD/Intel via the
  Vivado / Quartus vendor backends are in; the netlist-level equivalence prover
  miters RTL against the vendor post-impl netlist (`prove --netlist`), and
  bitstream reconstruction is pluggable across IceStorm, Project X-Ray (AMD
  7-series) and Project Apicula (Gowin) — install the decoding tools and those
  parts lift to the bitstream tier automatically.
- **Better models**: richer netlist/graph features, per-path timing prediction.
- **Real RTL transforms**: automatic pipelining, high-fanout replication, arithmetic
  rebalancing, memory rewriting for better BRAM inference.
- **Stage replacement**: swap individual vendor stages once our methods consistently win.
