"""行为克隆热启动（issue #6 训练准备）的单元测试。"""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ipd.agents.baseline import baseline_agent, strategy as baseline_strategy
    from ipd.pool import make_agent
    from ipd.sim import Agent, TournamentConfig, run_tournament
    from ipd.sim import rules
    from ipd.training.imitation import (
        BaselineTeacher,
        bc_checkpoint_payload,
        collect_demonstrations,
        collect_student_rollouts,
        label_episodes_with_teacher,
        train_behavior_cloning,
    )
    from ipd.training.policy import IPDPolicy
    from ipd.training.rl_train import load_checkpoint, save_checkpoint


def tiny_config(rounds=1):
    return TournamentConfig(rounds=rounds, hide_prob=0.0, timeout=None)


@unittest.skipIf(torch is None, "torch not installed")
class BaselineTeacherTest(unittest.TestCase):
    def test_teacher_matches_baseline_on_synthetic_batch(self):
        torch.manual_seed(0)
        batch, length = 4, 5
        own = torch.randint(0, 3, (batch, 20, 25))
        opp = torch.randint(0, 3, (batch, 20, 25))
        my = torch.randint(1, 3, (batch, length))
        their = torch.randint(1, 3, (batch, length))
        rs = torch.tensor([0, 3, 7, 2])
        teacher = BaselineTeacher()
        actions, probs, values = teacher.act_with_value(
            own, opp, my, their, rs, length
        )
        self.assertEqual(actions.shape, (batch,))
        self.assertEqual(probs.shape, (batch, 2))
        self.assertEqual(values.shape, (batch,))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(batch)))
        self.assertTrue(set(int(value) for value in actions) <= {0, 1})
        for index in range(batch):
            expected = baseline_strategy(
                int(rs[index]),
                length,
                1,
                tuple("c" if value == 1 else "b" for value in my[index].tolist()),
                2,
                tuple("c" if value == 1 else "b" for value in their[index].tolist()),
                rules.REPAY,
                tuple(
                    tuple("mcb"[int(value)] for value in row)
                    for row in own[index].tolist()
                ),
                tuple(
                    tuple("mcb"[int(value)] for value in row)
                    for row in opp[index].tolist()
                ),
            )
            self.assertEqual(int(actions[index]), 0 if expected == "c" else 1)

    def test_teacher_empty_history_opens_cooperatively(self):
        teacher = BaselineTeacher()
        empty = torch.zeros((1, 20, 25), dtype=torch.long)
        my = torch.zeros((1, 0), dtype=torch.long)
        their = torch.zeros((1, 0), dtype=torch.long)
        actions, _, _ = teacher.act_with_value(
            empty, empty, my, their, torch.tensor([0]), 0
        )
        self.assertEqual(int(actions[0]), 0)


@unittest.skipIf(torch is None, "torch not installed")
class DemonstrationsTest(unittest.TestCase):
    def test_teacher_trajectories_match_simulator_scores(self):
        cfg = tiny_config(rounds=1)
        opponents = [make_agent("all_c"), make_agent("all_d"), make_agent("tft")]
        episodes = collect_demonstrations(
            opponents,
            cfg,
            seasons=1,
            seed=0,
            opponents_per_round=3,
        )
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertEqual(len(episode.matches), 3)
        self.assertEqual(episode.transitions, 3 * rules.TURNS)
        for match in episode.matches:
            opponent = next(a for a in opponents if a.name == match.opponent)
            baseline = baseline_agent()
            result = run_tournament([baseline, opponent], seed=0, cfg=cfg)
            self.assertEqual(match.score, result.scores[0], match.opponent)

    def test_collect_and_train_bc(self):
        torch.manual_seed(0)
        cfg = tiny_config(rounds=1)
        opponents = [make_agent("all_c"), make_agent("all_d")]
        episodes = collect_demonstrations(
            opponents, cfg, seasons=1, seed=0, opponents_per_round=2
        )
        policy = IPDPolicy()
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        metrics = train_behavior_cloning(
            policy,
            episodes,
            epochs=1,
            lr=1e-3,
            batch_size=32,
            seed=0,
            verbose=False,
        )
        self.assertEqual(len(metrics), 1)
        self.assertTrue(0.0 <= metrics[0]["accuracy"] <= 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics[0]["loss"])))
        self.assertTrue(any(
            not torch.equal(parameter, old)
            for parameter, old in zip(policy.parameters(), before)
        ))

    def test_label_student_states_with_teacher(self):
        torch.manual_seed(0)
        cfg = tiny_config(rounds=1)
        opponents = [make_agent("all_c"), make_agent("all_d"), make_agent("tft")]
        policy = IPDPolicy()
        episodes = collect_student_rollouts(
            policy, opponents, cfg, seasons=1, seed=0, opponents_per_round=2
        )
        labeled = label_episodes_with_teacher(episodes, BaselineTeacher())
        self.assertEqual(labeled, 2 * rules.TURNS)
        for match in episodes[0].matches:
            self.assertEqual(len(match.labels), len(match.actions))
            self.assertTrue(all(label in (0, 1) for label in match.labels))
        metrics = train_behavior_cloning(
            policy, episodes, epochs=1, batch_size=32, seed=0, verbose=False
        )
        self.assertTrue(0.0 <= metrics[0]["accuracy"] <= 1.0)

    def test_bc_checkpoint_can_be_loaded(self):
        torch.manual_seed(0)
        cfg = tiny_config(rounds=1)
        episodes = collect_demonstrations(
            [make_agent("all_c"), make_agent("tft")],
            cfg,
            seasons=1,
            seed=0,
            opponents_per_round=2,
        )
        policy = IPDPolicy()
        metrics = train_behavior_cloning(
            policy, episodes, epochs=1, batch_size=32, seed=0, verbose=False
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bc.pt"
            save_checkpoint(path, bc_checkpoint_payload(policy, 1, metrics, 1))
            restored, payload = load_checkpoint(path)
            self.assertEqual(payload["train_config"]["method"], "behavior_cloning")
            self.assertEqual(
                restored.num_parameters(), policy.num_parameters()
            )
            policy.eval()
            restored.eval()
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


if __name__ == "__main__":
    unittest.main()
