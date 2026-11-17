"""训练循环、评估与阶段调度（issue #6，sub-issue 8.3–8.4）的单元测试。"""

from dataclasses import replace
import json
import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ipd.agents.baseline import baseline_agent
    from ipd.pool import make_agent
    from ipd.pool.registry import STAGE_HIDE_PROB, STAGE_LAYERS, agents_for_stage
    from ipd.sim import TournamentConfig
    from ipd.training.env import TrainingEnv
    from ipd.training.policy import IPDPolicy
    from ipd.training.rl_train import (
        TrainConfig,
        compute_returns,
        evaluate_policy,
        load_checkpoint,
        train_stage,
        update_policy,
    )
    from ipd.training.schedule import (
        CANONICAL_OPPONENTS,
        STAGE_SPECS,
        acceptance_report,
        stage_checkpoint_dir,
        stage_setup,
        stage_spec,
    )


def tiny_config(rounds=2):
    return TournamentConfig(rounds=rounds, hide_prob=0.0, timeout=None)


def tiny_episode(policy, rounds=2, seed=0):
    opponents = [make_agent("all_c"), make_agent("all_d"), make_agent("tft")]
    env = TrainingEnv(opponents, cfg=tiny_config(rounds), seed=seed, opponents_per_round=2)
    env.reset(seed=seed)
    policy.eval()
    generator = torch.Generator().manual_seed(seed)
    return env.rollout(policy, generator=generator)


@unittest.skipIf(torch is None, "torch not installed")
class ReturnTest(unittest.TestCase):
    def test_compute_returns(self):
        self.assertEqual(compute_returns([1.0, 2.0, 3.0]), [6.0, 5.0, 3.0])
        self.assertEqual(
            compute_returns([1.0, 2.0, 3.0], gamma=0.5), [2.75, 3.5, 3.0]
        )
        self.assertEqual(compute_returns([]), [])
        with self.assertRaises(ValueError):
            compute_returns([1.0], gamma=1.5)


@unittest.skipIf(torch is None, "torch not installed")
class UpdateTest(unittest.TestCase):
    def test_update_changes_parameters_and_returns_metrics(self):
        torch.manual_seed(0)
        policy = IPDPolicy()
        episode = tiny_episode(policy)
        from ipd.training.auxiliary import NextMovePredictor

        predictor = NextMovePredictor(policy.embed_dim)
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(predictor.parameters()), lr=1e-3
        )
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        metrics = update_policy(
            policy,
            predictor,
            optimizer,
            episode,
            TrainConfig(batch_size=64),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertGreater(metrics["transitions"], 0)
        self.assertTrue(all(
            torch.isfinite(torch.tensor(metrics[key]))
            for key in ("loss", "policy_loss", "value_loss", "entropy", "next_move_loss")
        ))
        self.assertGreaterEqual(metrics["entropy"], 0.0)
        self.assertTrue(any(
            not torch.equal(parameter, old)
            for parameter, old in zip(policy.parameters(), before)
        ))

    def test_update_with_imitation_loss(self):
        torch.manual_seed(2)
        policy = IPDPolicy()
        episode = tiny_episode(policy, rounds=1)
        for match in episode.matches:
            match.labels = [0] * len(match.actions)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = update_policy(
            policy,
            None,
            optimizer,
            episode,
            TrainConfig(
                batch_size=32,
                next_move_coef=0.0,
                contrastive_coef=0.0,
                imitation_coef=0.5,
            ),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertGreater(metrics["imitation_loss"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_update_without_auxiliary_losses(self):
        torch.manual_seed(1)
        policy = IPDPolicy()
        episode = tiny_episode(policy, rounds=1)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = update_policy(
            policy,
            None,
            optimizer,
            episode,
            TrainConfig(batch_size=32, next_move_coef=0.0, contrastive_coef=0.0),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(metrics["next_move_loss"], 0.0)
        self.assertEqual(metrics["contrastive_loss"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))


@unittest.skipIf(torch is None, "torch not installed")
class EvaluationTest(unittest.TestCase):
    def test_evaluate_policy_reports_baseline_and_regret(self):
        torch.manual_seed(0)
        policy = IPDPolicy().eval()
        opponents = agents_for_stage(1, seed=0)
        result = evaluate_policy(
            policy,
            opponents,
            tiny_config(rounds=1),
            seeds=(0,),
            baseline=baseline_agent(),
        )
        for key in ("policy_per_match", "baseline_per_match", "rank", "faults"):
            self.assertIn(key, result)
        self.assertEqual(result["n_agents"], len(opponents) + 2)
        self.assertEqual(result["faults"], 0)
        self.assertIsInstance(result["policy_per_match"], float)
        self.assertGreaterEqual(result["rank"], 1)
        self.assertEqual(set(result["regret"]), set(CANONICAL_OPPONENTS))
        for name in CANONICAL_OPPONENTS:
            self.assertIn(name, result["per_opponent"])

    def test_evaluate_policy_without_baseline(self):
        torch.manual_seed(0)
        policy = IPDPolicy().eval()
        result = evaluate_policy(
            policy,
            [make_agent("all_c"), make_agent("tft")],
            tiny_config(rounds=1),
            seeds=(0,),
        )
        self.assertIsNone(result["baseline_per_match"])
        self.assertEqual(result["regret"], {})

    def test_hide_robustness(self):
        torch.manual_seed(0)
        policy = IPDPolicy().eval()
        config = TournamentConfig(rounds=1, hide_prob=0.5, timeout=None)
        result = evaluate_policy(
            policy,
            [make_agent("all_c"), make_agent("tft")],
            config,
            seeds=(0,),
            include_hide_robustness=True,
        )
        self.assertIn("hide_drop_ratio", result)
        self.assertIn("clear_per_match", result)


@unittest.skipIf(torch is None, "torch not installed")
class TrainStageTest(unittest.TestCase):
    def test_smoke_train_stage_with_checkpoint(self):
        torch.manual_seed(0)
        policy = IPDPolicy()
        opponents = [make_agent("all_c"), make_agent("all_d")]
        with tempfile.TemporaryDirectory() as tmp:
            history = train_stage(
                1,
                policy,
                opponents,
                tiny_config(rounds=1),
                baseline=None,
                num_episodes=2,
                train_config=TrainConfig(batch_size=32, seed=0),
                checkpoint_dir=tmp,
                eval_every=0,
                verbose=False,
            )
            self.assertEqual(len(history), 2)
            checkpoint = Path(tmp) / "stage1_last.pt"
            self.assertTrue(checkpoint.exists())
            restored, payload = load_checkpoint(checkpoint)
            self.assertEqual(payload["stage"], 1)
            self.assertEqual(payload["iteration"], 1)
            restored.eval()
            policy.eval()
            args = (
                torch.zeros(1, 20, 25, dtype=torch.long),
                torch.zeros(1, 20, 25, dtype=torch.long),
                torch.zeros(1, 0, dtype=torch.long),
                torch.zeros(1, 0, dtype=torch.long),
                torch.tensor([0]),
                0,
            )
            with torch.no_grad():
                self.assertTrue(torch.allclose(policy(*args), restored(*args)))
            # 断点续训：恢复预测头/优化器状态后能继续跑一个赛季。
            resumed_history = train_stage(
                1,
                restored,
                opponents,
                tiny_config(rounds=1),
                num_episodes=1,
                train_config=TrainConfig(batch_size=32, seed=0),
                eval_every=0,
                verbose=False,
                resume_payload=payload,
            )
            self.assertEqual(len(resumed_history), 1)

    def test_train_stage_writes_json_log(self):
        torch.manual_seed(0)
        policy = IPDPolicy()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "metrics.jsonl"
            train_stage(
                1,
                policy,
                [make_agent("tft")],
                tiny_config(rounds=1),
                num_episodes=1,
                train_config=TrainConfig(batch_size=32, seed=0),
                eval_every=0,
                log_path=log,
                verbose=False,
            )
            lines = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["iteration"], 0)
            self.assertIn("loss", payload)


@unittest.skipIf(torch is None, "torch not installed")
class ScheduleTest(unittest.TestCase):
    def test_stage_specs_follow_adr_0006(self):
        for stage in (1, 2, 3, 4):
            with self.subTest(stage=stage):
                spec = stage_spec(stage)
                self.assertEqual(spec.stage, stage)
                self.assertEqual(spec.layers, STAGE_LAYERS[stage])
                self.assertEqual(spec.hide_prob, STAGE_HIDE_PROB[stage])
        self.assertTrue(STAGE_SPECS[4].self_play)
        with self.assertRaises(ValueError):
            stage_spec(5)

    def test_stage_setup_returns_registry_pool_and_config(self):
        for stage in (1, 2, 3, 4):
            with self.subTest(stage=stage):
                spec, agents, config = stage_setup(stage, seed=0)
                self.assertEqual(len(agents), len(agents_for_stage(stage, seed=0)))
                self.assertEqual(config.hide_prob, STAGE_HIDE_PROB[stage])
                self.assertEqual(config.hide_scope, "match")
                self.assertEqual(spec.stage, stage)

    def test_gate_stage1(self):
        metrics = {
            "policy_per_match": 50.0,
            "baseline_per_match": 45.0,
            "regret": {name: 0.0 for name in CANONICAL_OPPONENTS},
        }
        result = acceptance_report(1, metrics)
        self.assertTrue(result.passed)
        failed = acceptance_report(1, {**metrics, "policy_per_match": 40.0})
        self.assertFalse(failed.passed)
        worse = acceptance_report(
            1, {**metrics, "regret": {name: 1.0 for name in CANONICAL_OPPONENTS}}
        )
        self.assertFalse(worse.passed)

    def test_gate_stage2_compares_previous_regret(self):
        metrics = {
            "policy_per_match": 46.0,
            "baseline_per_match": 45.0,
            "regret": {name: 0.4 for name in CANONICAL_OPPONENTS},
        }
        previous = {
            "policy_per_match": 45.0,
            "baseline_per_match": 45.0,
            "regret": {name: 0.0 for name in CANONICAL_OPPONENTS},
        }
        self.assertTrue(acceptance_report(2, metrics, previous=previous).passed)
        worse = {
            **metrics,
            "regret": {name: 2.0 for name in CANONICAL_OPPONENTS},
        }
        self.assertFalse(acceptance_report(2, worse, previous=previous).passed)

    def test_gate_stage3_hide_robustness(self):
        metrics = {
            "policy_per_match": 50.0,
            "baseline_per_match": 45.0,
            "regret": {name: 0.0 for name in CANONICAL_OPPONENTS},
        }
        self.assertTrue(acceptance_report(3, metrics).passed)
        self.assertIn(
            "未测量", " ".join(acceptance_report(3, metrics).notes)
        )
        failed = acceptance_report(3, {**metrics, "hide_drop_ratio": 0.5})
        self.assertFalse(failed.passed)

    def test_gate_requires_scores(self):
        with self.assertRaises(ValueError):
            acceptance_report(1, {})
        with self.assertRaises(ValueError):
            stage_spec(0)
        with self.assertRaises(ValueError):
            stage_spec("x")

    def test_checkpoint_dir(self):
        self.assertEqual(
            stage_checkpoint_dir(Path("runs"), 2), Path("runs/stage2")
        )


if __name__ == "__main__":
    unittest.main()
