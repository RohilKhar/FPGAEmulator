from fpgaforge.backends.base import FlowOptions, RunMetrics, RunResult
from fpgaforge.corpus import Corpus


def _make_result(fmax: float, ok: bool = True) -> RunResult:
    return RunResult(
        design_id="demo:abc123",
        options=FlowOptions(retime=True),
        metrics=RunMetrics(fmax_mhz=fmax, routed_ok=ok, luts=100),
        features={"num_carries": 10.0},
        success=ok,
        backend="mock",
    )


def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "corpus.jsonl"
    corpus = Corpus(path)
    corpus.append(_make_result(120.0))
    corpus.append(_make_result(0.0, ok=False))

    rows = corpus.load()
    assert len(rows) == 2
    assert len(corpus) == 2
    assert rows[0]["metrics"]["fmax_mhz"] == 120.0
    assert rows[0]["options"]["retime"] is True


def test_training_rows_filters_failures(tmp_path):
    path = tmp_path / "corpus.jsonl"
    corpus = Corpus(path)
    corpus.append(_make_result(120.0))
    corpus.append(_make_result(0.0, ok=False))

    train = corpus.training_rows()
    assert len(train) == 1
    assert train[0]["metrics"]["fmax_mhz"] == 120.0
