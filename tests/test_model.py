from fpgaforge.backends.base import FlowOptions
from fpgaforge.model import FmaxPredictor, ModelReport, make_vector


def _row(fmax, carries, retime, luts=None, routed=True, success=True):
    return {
        "success": success,
        "metrics": {
            "fmax_mhz": fmax,
            "routed_ok": routed,
            "luts": luts if luts is not None else int(500 + carries * 10),
        },
        "features": {"num_carries": carries, "max_fanout": 8.0, "num_dsp": 1.0},
        "options": FlowOptions(retime=retime).to_dict(),
    }


def test_heuristic_rewards_good_knobs():
    predictor = FmaxPredictor()  # untrained -> heuristic
    feats = {"num_carries": 20.0, "max_fanout": 10.0, "num_dsp": 1.0}
    base = predictor.predict(feats, FlowOptions())
    tuned = predictor.predict(feats, FlowOptions(retime=True, abc9=True))
    assert tuned > base


def test_fit_trains_with_enough_samples():
    rows = []
    for i in range(20):
        carries = 5 + i
        retime = i % 2 == 0
        fmax = 200.0 - carries + (15.0 if retime else 0.0)
        rows.append(_row(fmax, carries, retime))

    predictor = FmaxPredictor().fit(rows)
    assert predictor.trained is True
    assert predictor.n_samples == 20
    pred = predictor.predict(rows[0]["features"], FlowOptions(retime=True))
    assert pred > 0


def test_fit_falls_back_when_too_few():
    rows = [_row(150.0, 10, True) for _ in range(3)]
    predictor = FmaxPredictor().fit(rows)
    assert predictor.trained is False


def test_save_load_roundtrip(tmp_path):
    rows = [_row(180.0 - i, 5 + i, i % 2 == 0) for i in range(20)]
    predictor = FmaxPredictor().fit(rows)
    path = tmp_path / "model.joblib"
    predictor.save(path)

    loaded = FmaxPredictor.load(path)
    assert loaded.trained is True
    assert loaded.n_samples == 20


def test_make_vector_length():
    from fpgaforge import features as feat

    vec = make_vector(feat.empty_features(), FlowOptions())
    assert len(vec) == len(feat.FEATURE_NAMES) + 5


def _learnable_rows(n=40):
    rows = []
    for i in range(n):
        carries = 5 + (i % 20)
        retime = i % 2 == 0
        fmax = 200.0 - carries + (15.0 if retime else 0.0)
        rows.append(_row(fmax, carries, retime))
    return rows


def test_evaluate_reports_cross_validated_skill():
    report = FmaxPredictor.evaluate(_learnable_rows())
    assert isinstance(report, ModelReport)
    assert report.trained is True
    assert report.n == 40
    # On this near-linear target the model must beat the mean-baseline.
    assert report.mae < report.baseline_mae
    assert report.within_10pct > 0.5
    assert report.importances  # non-empty feature importances
    # num_carries should carry meaningful signal.
    assert report.importances.get("num_carries", 0.0) > 0.0


def test_evaluate_too_few_samples_is_untrained():
    report = FmaxPredictor.evaluate([_row(150.0, 10, True) for _ in range(4)])
    assert report.trained is False
    assert report.n == 4


def test_lut_head_predicts_resources():
    predictor = FmaxPredictor().fit(_learnable_rows())
    assert predictor.lut_model is not None
    pred = predictor.predict_luts({"num_carries": 15.0, "num_luts": 0.0}, FlowOptions())
    assert pred > 0.0


def test_routability_head_learns_failures():
    rows = _learnable_rows(30)
    # Add clearly-failing points (huge design that does not route).
    for i in range(15):
        rows.append(
            _row(0.0, carries=200 + i, retime=False, luts=99999, routed=False, success=False)
        )
    predictor = FmaxPredictor().fit(rows)
    assert predictor.route_model is not None
    p_small = predictor.routed_probability({"num_carries": 10.0}, FlowOptions())
    p_huge = predictor.routed_probability({"num_carries": 210.0}, FlowOptions())
    assert p_small > p_huge


def test_untrained_routability_is_optimistic():
    predictor = FmaxPredictor()
    assert predictor.routed_probability({}, FlowOptions()) == 1.0


def test_save_load_roundtrip_heads(tmp_path):
    rows = _learnable_rows(30)
    for i in range(12):
        rows.append(_row(0.0, 300 + i, False, luts=99999, routed=False, success=False))
    predictor = FmaxPredictor().fit(rows)
    path = tmp_path / "m.joblib"
    predictor.save(path)
    loaded = FmaxPredictor.load(path)
    assert loaded.lut_model is not None
    assert loaded.route_model is not None
