from fpgaforge.backends.mock import MockBackend
from fpgaforge.corpus import Corpus
from fpgaforge.optimizer import candidate_space, optimize

MAC_RTL = """
module mac(input clk, input rst, input [15:0] a, input [15:0] b,
           input [31:0] c, output reg [31:0] y);
  always @(posedge clk) begin
    if (rst) y <= 0;
    else y <= a * b + c;
  end
endmodule
"""


def _write_rtl(tmp_path):
    p = tmp_path / "mac.v"
    p.write_text(MAC_RTL)
    return p


def test_candidate_space_is_deduped():
    cands = candidate_space()
    keys = [c.key() for c in cands]
    assert len(keys) == len(set(keys))
    assert len(cands) > 1


def test_optimize_improves_or_matches_baseline(tmp_path):
    rtl = _write_rtl(tmp_path)
    corpus = Corpus(tmp_path / "corpus.jsonl")
    result = optimize(
        rtl=str(rtl),
        top="mac",
        iterations=8,
        backend=MockBackend(),
        corpus=corpus,
        run_dir=tmp_path / "runs",
    )

    assert result.backend == "mock"
    assert result.baseline is not None
    assert result.best is not None
    # Optimizer should never do worse than the baseline knobs.
    assert result.best.metrics.fmax_mhz >= result.baseline.metrics.fmax_mhz
    assert result.improvement_pct >= 0.0
    # Every run is logged to the corpus.
    assert len(corpus.load()) == len(result.history)
    assert len(result.history) == 8


def test_optimize_records_bitstream_or_metrics(tmp_path):
    rtl = _write_rtl(tmp_path)
    result = optimize(
        rtl=str(rtl),
        top="mac",
        iterations=4,
        backend=MockBackend(),
        corpus=Corpus(tmp_path / "corpus.jsonl"),
        run_dir=tmp_path / "runs",
    )
    assert result.best.metrics.luts > 0
    assert "maximize_fmax" == result.objective
