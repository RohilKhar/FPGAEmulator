from fpgaforge.backends.mock import MockBackend
from fpgaforge.bootstrap import BootstrapSpec, bootstrap_corpus
from fpgaforge.corpus import Corpus
from fpgaforge.model import FmaxPredictor

COUNTER = """
module counter(input clk, input rst, output reg [7:0] q);
  always @(posedge clk) if (rst) q <= 0; else q <= q + 1;
endmodule
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_bootstrap_populates_corpus(tmp_path):
    rtl = _write(tmp_path, "counter.v", COUNTER)
    corpus = Corpus(tmp_path / "corpus.jsonl")
    spec = BootstrapSpec(rtl_files=(rtl,), top="counter")

    report = bootstrap_corpus(
        [spec],
        seeds=(1, 2),
        backend=MockBackend(),
        corpus=corpus,
        run_dir=tmp_path / "runs",
    )

    assert report.runs > 0
    assert report.runs == report.corpus_size
    assert len(corpus.load()) == report.runs


def test_bootstrap_multiplies_over_targets_and_clocks(tmp_path):
    rtl = _write(tmp_path, "counter.v", COUNTER)
    corpus = Corpus(tmp_path / "corpus.jsonl")
    spec = BootstrapSpec(
        rtl_files=(rtl,),
        top="counter",
        targets=("ice40_up5k", "ecp5_25k"),
        clocks_ns=(10.0, 5.0),
    )

    report = bootstrap_corpus(
        [spec],
        seeds=(1,),
        backend=MockBackend(),
        corpus=corpus,
        run_dir=tmp_path / "runs",
        max_per_design=3,
    )

    # 2 targets x 2 clocks = 4 design combos.
    assert report.designs == 4
    assert report.runs == report.points


def test_bootstrapped_corpus_is_trainable(tmp_path):
    rtl = _write(tmp_path, "counter.v", COUNTER)
    corpus = Corpus(tmp_path / "corpus.jsonl")
    spec = BootstrapSpec(
        rtl_files=(rtl,),
        top="counter",
        clocks_ns=(12.0, 10.0, 8.0, 6.0),
    )
    bootstrap_corpus(
        [spec], seeds=(1, 2, 3), backend=MockBackend(), corpus=corpus,
        run_dir=tmp_path / "runs",
    )
    rows = corpus.load()
    predictor = FmaxPredictor().fit(rows)
    # MockBackend is deterministic per knob set, so plenty of usable samples.
    assert predictor.n_samples == len(
        [r for r in rows if r["success"] and r["metrics"]["routed_ok"]]
    )
