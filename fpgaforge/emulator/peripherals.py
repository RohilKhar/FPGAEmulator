"""Virtual board / peripheral models for the bitstream-level emulator.

The fabric emulator reconstructs *what the flashed bitstream computes*. A real
FPGA, though, is soldered to a board: an oscillator drives its clock, a reset
controller wakes it, and its pins talk to LEDs, a UART, buttons, switches, and
GPIO. This module models that board so the reconstructed bitstream can be *run*
against behavioral peripheral models -- you watch LEDs blink and read the text a
UART actually transmits, straight out of the bits you would flash.

Everything here is pure Verilog/text generation plus log parsing, so the
peripheral models and their decoders are unit-testable without any tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..virtual.board import Port, detect_clock, detect_reset, _reset_is_active_high

# Peripheral roles a top-level pin can play on the virtual board.
CLOCK = "clock"
RESET = "reset"
LED = "led"
UART_TX = "uart_tx"      # DUT output -> board receives/decodes characters
UART_RX = "uart_rx"      # DUT input  -> board transmits characters into the DUT
BUTTON = "button"
SWITCH = "switch"
GPIO_OUT = "gpio_out"
GPIO_IN = "gpio_in"


@dataclass
class BoardConfig:
    """How the virtual board is wired and driven."""

    clock_mhz: float = 12.0
    baud: int = 1_000_000           # must match the DUT's bit rate at clock_mhz
    duration_us: float = 60.0       # how long to run the board
    reset_cycles: int = 8
    reset_active_high: bool | None = None
    clock: str | None = None
    reset: str | None = None
    # Bytes the board's UART transmitter sends *into* the DUT's rx pin.
    uart_rx_bytes: list[int] = field(default_factory=list)
    # Static input levels (e.g. {"sw": 3}); anything unset is tied low.
    input_levels: dict[str, int] = field(default_factory=dict)
    # Button presses: name -> list of (time_us, value).
    button_schedule: dict[str, list[tuple[float, int]]] = field(default_factory=dict)
    # Explicit pin-role overrides, e.g. {"io3": "uart_tx"}.
    pin_roles: dict[str, str] = field(default_factory=dict)
    vcd_path: str | None = "board.vcd"


def classify_pins(ports: list[Port], cfg: BoardConfig) -> dict[str, str]:
    """Map each top-level pin to a peripheral role (name heuristics + overrides)."""
    clk = detect_clock(ports, cfg.clock)
    rst = detect_reset(ports, cfg.reset)
    roles: dict[str, str] = {}
    for p in ports:
        if p.name in cfg.pin_roles:
            roles[p.name] = cfg.pin_roles[p.name]
            continue
        if clk and p.name == clk.name:
            roles[p.name] = CLOCK
            continue
        if rst and p.name == rst.name:
            roles[p.name] = RESET
            continue
        roles[p.name] = _role_from_name(p)
    return roles


def _role_from_name(p: Port) -> str:
    low = p.name.lower()
    if p.direction in ("output", "inout"):
        if "led" in low:
            return LED
        if re.search(r"(^|_)(tx|txd|uart_tx|ser_tx|tx_o)($|_)", low) or low.endswith("tx"):
            return UART_TX
        return GPIO_OUT
    # inputs
    if re.search(r"(^|_)(rx|rxd|uart_rx|ser_rx)($|_)", low) or low.endswith("rx"):
        return UART_RX
    if "btn" in low or "button" in low or low.startswith("key"):
        return BUTTON
    if low.startswith("sw") or "switch" in low or "dip" in low:
        return SWITCH
    return GPIO_IN


# ---------------------------------------------------------------------- #
def render_board_tb(
    top: str, ports: list[Port], cfg: BoardConfig, roles: dict[str, str]
) -> str:
    """Render a Verilog testbench that wires `top` to behavioral peripherals."""
    clk = detect_clock(ports, cfg.clock)
    if clk is None:
        raise ValueError("no clock port detected; the virtual board needs a clock")
    rst = detect_reset(ports, cfg.reset)
    active_high = _reset_is_active_high(rst, cfg) if rst else True
    assert_val = "1'b1" if active_high else "1'b0"
    deassert_val = "1'b0" if active_high else "1'b1"

    half_ns = 500.0 / cfg.clock_mhz          # half clock period in ns
    bit_ns = 1_000_000_000.0 / cfg.baud      # UART bit period in ns
    duration_ns = cfg.duration_us * 1000.0

    L: list[str] = ["`timescale 1ns/1ps",
                    "// Auto-generated fpgaforge virtual-board harness.",
                    "module board_tb;"]
    L.append(f"  localparam real BIT_NS = {bit_ns:.4f};")

    # ---- declarations ----
    L.append(f"  reg {clk.name};")
    if rst:
        L.append(f"  reg {rst.name};")
    for p in ports:
        role = roles.get(p.name)
        if p.name in (clk.name, getattr(rst, "name", None)):
            continue
        rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
        if p.direction == "input":
            L.append(f"  reg {rng}{p.name};")
        else:
            L.append(f"  wire {rng}{p.name};")

    # ---- DUT ----
    conns = ", ".join(f".{p.name}({p.name})" for p in ports)
    L.append(f"  {top} dut ({conns});")

    # ---- clock oscillator ----
    L.append(f"  initial {clk.name} = 1'b0;")
    L.append(f"  always #{half_ns:g} {clk.name} = ~{clk.name};")

    # ---- VCD ----
    if cfg.vcd_path:
        L.append("  initial begin")
        L.append(f'    $dumpfile("{cfg.vcd_path}");')
        L.append("    $dumpvars(0, board_tb);")
        L.append("  end")

    # ---- reset controller + static input levels ----
    L.append("  initial begin")
    if rst:
        L.append(f"    {rst.name} = {assert_val};")
    for p in ports:
        if p.direction != "input" or p.name in (clk.name, getattr(rst, "name", None)):
            continue
        role = roles.get(p.name)
        if role == UART_RX:
            L.append(f"    {p.name} = 1'b1;")   # UART idle-high
        else:
            level = cfg.input_levels.get(p.name, 0)
            L.append(f"    {p.name} = {p.width}'d{level};")
    if rst:
        L.append(f"    repeat ({cfg.reset_cycles}) @(posedge {clk.name});")
        L.append(f"    {rst.name} = {deassert_val};")
    L.append("  end")

    # ---- peripheral processes ----
    for p in ports:
        role = roles.get(p.name)
        if role == LED:
            L += _led_monitor(p)
        elif role == GPIO_OUT:
            L += _gpio_monitor(p)
        elif role == UART_TX:
            L += _uart_rx_model(p)          # board *receives* the DUT's tx
        elif role == UART_RX and cfg.uart_rx_bytes:
            L += _uart_tx_model(p, cfg.uart_rx_bytes, cfg.reset_cycles, half_ns)
        elif role == BUTTON and p.name in cfg.button_schedule:
            L += _button_driver(p, cfg.button_schedule[p.name])

    # ---- run window ----
    L.append("  initial begin")
    L.append(f"    #{duration_ns:g};")
    L.append('    $display("BOARD_DONE");')
    L.append("    $finish;")
    L.append("  end")
    L.append("endmodule")
    return "\n".join(L) + "\n"


def _sig(p: Port) -> str:
    """A 1-bit view of a pin (bit 0 for buses) for serial peripherals."""
    return f"{p.name}[0]" if p.width > 1 else p.name


def _led_monitor(p: Port) -> list[str]:
    val = p.name
    return [
        f"  initial $display(\"LED {p.name} %0t %0d\", $time, {val});",
        f"  always @({p.name}) $display(\"LED {p.name} %0t %0d\", $time, {val});",
    ]


def _gpio_monitor(p: Port) -> list[str]:
    return [
        f"  always @({p.name}) $display(\"GPIO {p.name} %0t %0d\", $time, {p.name});",
    ]


def _uart_rx_model(p: Port) -> list[str]:
    """Board-side UART receiver: decode characters the DUT transmits on `p`."""
    sig = _sig(p)
    rx = f"_rx_{p.name}"
    k = f"_k_{p.name}"
    return [
        f"  reg [7:0] {rx};",
        f"  integer {k};",
        "  initial begin",
        "    forever begin",
        f"      @(negedge {sig});",       # start bit
        "      #(BIT_NS * 1.5);",          # center of bit 0
        f"      for ({k} = 0; {k} < 8; {k} = {k} + 1) begin",
        f"        {rx}[{k}] = {sig};",
        "        #(BIT_NS);",
        "      end",
        f"      $display(\"UARTCHAR {p.name} %0d\", {rx});",
        "    end",
        "  end",
    ]


def _uart_tx_model(p: Port, data: list[int], reset_cycles: int, half_ns: float) -> list[str]:
    """Board-side UART transmitter: send `data` bytes into the DUT input `p`."""
    sig = p.name
    settle = (reset_cycles + 4) * half_ns * 2
    lines = ["  initial begin", f"    {sig} = 1'b1;", f"    #{settle:g};"]
    for byte in data:
        b = byte & 0xFF
        lines.append(f"    {sig} = 1'b0; #(BIT_NS);")            # start
        for i in range(8):
            lines.append(f"    {sig} = 1'b{(b >> i) & 1}; #(BIT_NS);")
        lines.append(f"    {sig} = 1'b1; #(BIT_NS);")            # stop
    lines.append("  end")
    return lines


def _button_driver(p: Port, schedule: list[tuple[float, int]]) -> list[str]:
    lines = ["  initial begin"]
    prev = 0.0
    for t_us, val in schedule:
        delay = max(0.0, t_us * 1000.0 - prev)
        lines.append(f"    #{delay:g}; {p.name} = {p.width}'d{val};")
        prev = t_us * 1000.0
    lines.append("  end")
    return lines


# ---------------------------------------------------------------------- #
@dataclass
class UartCapture:
    pin: str
    bytes_: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(chr(b) if 9 <= b < 127 else f"\\x{b:02x}" for b in self.bytes_)


@dataclass
class BoardResult:
    design_id: str = ""
    ran: bool = False
    roles: dict[str, str] = field(default_factory=dict)
    uart: dict[str, UartCapture] = field(default_factory=dict)
    led_events: list[tuple[str, int, int]] = field(default_factory=list)  # (pin, t_ns, val)
    gpio_events: list[tuple[str, int, int]] = field(default_factory=list)
    bitstream_path: str | None = None
    fabric = None
    vcd_path: str | None = None
    workdir: str | None = None
    log: str = ""
    error: str | None = None

    def led_final(self) -> dict[str, int]:
        final: dict[str, int] = {}
        for pin, _t, val in self.led_events:
            final[pin] = val
        return final

    def led_toggle_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        last: dict[str, int] = {}
        for pin, _t, val in self.led_events:
            if pin in last and last[pin] != val:
                counts[pin] = counts.get(pin, 0) + 1
            last[pin] = val
        return counts

    def summary(self) -> str:
        if self.error:
            return f"virtual board: ERROR\ndesign : {self.design_id}\nerror  : {self.error}"
        lines = ["virtual board: RAN", f"design : {self.design_id}"]
        if self.bitstream_path:
            lines.append(f"bitstream: {self.bitstream_path}")
        # Peripherals wired.
        wired = sorted({r for n, r in self.roles.items() if r not in (CLOCK, RESET)})
        lines.append(f"peripherals: {', '.join(wired) if wired else 'none detected'}")
        for pin, cap in self.uart.items():
            lines.append(f"uart[{pin}]: \"{cap.text}\" ({len(cap.bytes_)} bytes)")
        toggles = self.led_toggle_counts()
        for pin, val in self.led_final().items():
            n = toggles.get(pin, 0)
            lines.append(f"led[{pin}]: {n} toggle(s), final={val}")
        if self.gpio_events:
            last = {}
            for pin, _t, v in self.gpio_events:
                last[pin] = v
            for pin, v in last.items():
                lines.append(f"gpio[{pin}]: final={v}")
        if self.vcd_path:
            lines.append(f"waveform: {self.vcd_path}")
        return "\n".join(lines)


def parse_board_log(output: str, roles: dict[str, str]) -> BoardResult:
    """Turn raw simulator output into a structured BoardResult (pure)."""
    res = BoardResult(roles=dict(roles), ran="BOARD_DONE" in output)
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("UARTCHAR "):
            _, pin, val = line.split()
            res.uart.setdefault(pin, UartCapture(pin=pin)).bytes_.append(int(val) & 0xFF)
        elif line.startswith("LED "):
            parts = line.split()
            res.led_events.append((parts[1], int(parts[2]), int(parts[3])))
        elif line.startswith("GPIO "):
            parts = line.split()
            res.gpio_events.append((parts[1], int(parts[2]), int(parts[3])))
    return res
