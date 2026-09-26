import json

from ddm_mcts.environments import ConnectFour
from ddm_mcts.evaluation.experiment import diagnostic_positions, run_experiment


def test_diagnostic_positions_are_legal_and_labeled_actions_are_available():
    env = ConnectFour()
    positions = diagnostic_positions(env)
    assert {p.name for p in positions} >= {"empty_board", "immediate_win", "immediate_block", "near_endgame"}
    for position in positions:
        assert not env.is_terminal(position.state)
        if position.correct_action is not None:
            assert position.correct_action in env.legal_actions(position.state)


def test_experiment_generates_reports_without_laya(tmp_path, monkeypatch):
    from ddm_mcts.evaluation import experiment
    from ddm_mcts.policies import laya_policy

    def unavailable(self):
        raise laya_policy.LayaUnavailableError("test runtime unavailable")

    monkeypatch.setattr(laya_policy.LayaPolicy, "_ensure_predictor", unavailable)
    monkeypatch.setattr(experiment, "_unit_tests", lambda: {"status": "pass", "output": "isolated", "return_code": 0, "latency_seconds": 0.0})
    output = run_experiment(games=1, budgets=[1], seed=7, results_root=tmp_path, skip_laya=True)
    assert (output / "summary.md").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["laya_available"] is False
    assert summary["bad_prior_recovery"]
    assert (output / "diagnostics" / "cache_stats.json").is_file()
