"""训练环境（issue #6，sub-issue 8.2）的单元测试。"""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ipd.pool import make_agent
    from ipd.sim import Agent, TournamentConfig, run_tournament
    from ipd.sim import rules
    from ipd.training.env import TrainingEnv
    from ipd.training.policy import IPDPolicy


class StubLabeler:
    """固定教师动作的标签器。"""

    def __init__(self, action):
        self.action = action

    def act_with_value(self, *encoder_inputs, **kwargs):
        batch = int(encoder_inputs[0].shape[0])
        actions = torch.full((batch,), self.action, dtype=torch.long)
        probs = torch.zeros((batch, 2))
        probs[:, self.action] = 1.0
        return actions, probs, torch.zeros(batch)


class StubPolicy:
    """固定动作的策略桩：只实现环境需要的 `act_with_value`。"""

    def __init__(self, action, with_value=True):
        self.action = action
        self.with_value = with_value

    def act_with_value(self, *encoder_inputs, deterministic=False, generator=None):
        batch = int(encoder_inputs[0].shape[0])
        probs = torch.zeros((batch, 2))
        probs[:, self.action] = 1.0
        actions = torch.full((batch,), self.action, dtype=torch.long)
        values = torch.zeros(batch) if self.with_value else None
        return actions, probs, values


def recorder(log):
    def strategy(*args):
        log.append(args)
        return "c"

    return strategy


@unittest.skipIf(torch is None, "torch not installed")
class TrainingEnvTest(unittest.TestCase):
    def test_rollout_matches_simulator(self):
        cfg = TournamentConfig(rounds=1, hide_prob=0.0, timeout=None)
        for action, reference in ((0, "all_c"), (1, "all_d")):
            with self.subTest(reference=reference):
                opponents = [
                    make_agent("all_c"),
                    make_agent("all_d"),
                    make_agent("tft"),
                    make_agent("grim"),
                ]
                env = TrainingEnv(opponents, cfg=cfg, seed=0, opponents_per_round=4)
                env.reset(seed=0)
                episode = env.rollout(StubPolicy(action))
                scores = {}
                for match in episode.matches:
                    scores[match.opponent] = match.score
                for opponent in opponents:
                    reference_agent = Agent(
                        f"{reference}#ref", make_agent(reference).fn
                    )
                    result = run_tournament(
                        [reference_agent, opponent], seed=0, cfg=cfg
                    )
                    self.assertEqual(scores[opponent.name], result.scores[0])

    def test_history_accumulates_across_rounds(self):
        log = []
        cfg = TournamentConfig(rounds=2, hide_prob=0.0, timeout=None)
        env = TrainingEnv(
            [Agent("rec", recorder(log))], cfg=cfg, seed=0, opponents_per_round=1
        )
        env.reset(seed=0)
        env.rollout(StubPolicy(0))
        self.assertEqual(len(log), 50)
        # 第 0 轮：对手看不到我方历史，全部补位。
        self.assertEqual(log[0][7], (rules.MISSING_ROW,) * 20)
        # 第 1 轮第一手：上一轮的记录出现在最后一行。
        second_round = log[25]
        view = second_round[7]
        self.assertEqual(view[:-1], (rules.MISSING_ROW,) * 19)
        self.assertEqual(view[-1], (rules.COOPERATE,) * 25)
        # 我方看到的对手历史（对手自己的动作 + 对手的对手的动作）。
        self.assertEqual(second_round[8][-1], (rules.COOPERATE,) * 25)

    def test_hide_prob_hides_history(self):
        log = []
        cfg = TournamentConfig(rounds=2, hide_prob=1.0, timeout=None)
        env = TrainingEnv(
            [Agent("rec", recorder(log))], cfg=cfg, seed=0, opponents_per_round=1
        )
        env.reset(seed=0)
        env.rollout(StubPolicy(0))
        self.assertEqual(log[25][7], (rules.MISSING_ROW,) * 20)
        self.assertEqual(log[25][8], (rules.MISSING_ROW,) * 20)

    def test_reproducible_with_seed(self):
        cfg = TournamentConfig(rounds=1, hide_prob=0.1, timeout=None)
        opponents = [make_agent("rand"), make_agent("tft"), make_agent("joss")]
        episodes = []
        for _ in range(2):
            torch.manual_seed(0)
            policy = IPDPolicy().eval()
            env = TrainingEnv(opponents, cfg=cfg, seed=7, opponents_per_round=3)
            env.reset(seed=7)
            generator = torch.Generator().manual_seed(42)
            episodes.append(env.rollout(policy, generator=generator))
        for first, second in zip(episodes[0].matches, episodes[1].matches):
            self.assertEqual(first.opponent, second.opponent)
            self.assertEqual(first.actions, second.actions)
            self.assertEqual(first.opp_moves, second.opp_moves)
            self.assertEqual(first.payoffs, second.payoffs)

    def test_shaping_penalises_unprovoked_defection(self):
        cfg = TournamentConfig(rounds=1, hide_prob=0.0, timeout=None)
        env = TrainingEnv([make_agent("all_c")], cfg=cfg, seed=0, opponents_per_round=1)
        env.reset(seed=0)
        plain = env.rollout(StubPolicy(1))
        env.reset(seed=0)
        shaped = env.rollout(StubPolicy(1), shaping=2.0)
        match_plain, match_shaped = plain.matches[0], shaped.matches[0]
        self.assertEqual(match_plain.payoffs, match_shaped.payoffs)
        # 第 0 手没有"上一手"，其余 24 手都算无端背叛。
        self.assertAlmostEqual(
            sum(match_shaped.rewards), sum(match_plain.rewards) - 2.0 * 24
        )

    def test_labeler_records_teacher_actions(self):
        cfg = TournamentConfig(rounds=1, hide_prob=0.0, timeout=None)
        env = TrainingEnv([make_agent("all_c")], cfg=cfg, seed=0, opponents_per_round=1)
        env.reset(seed=0)
        episode = env.rollout(StubPolicy(0), labeler=StubLabeler(1))
        self.assertEqual(episode.matches[0].labels, [1] * rules.TURNS)
        self.assertEqual(episode.matches[0].actions, [0] * rules.TURNS)

    def test_episode_shape_and_scores(self):
        cfg = TournamentConfig(rounds=3, hide_prob=0.0, timeout=None)
        opponents = [make_agent("all_c"), make_agent("all_d"), make_agent("tft")]
        env = TrainingEnv(opponents, cfg=cfg, seed=0, opponents_per_round=2)
        env.reset(seed=0)
        episode = env.rollout(StubPolicy(0))
        self.assertEqual(len(episode.matches), 6)
        self.assertEqual(episode.transitions, 6 * rules.TURNS)
        self.assertAlmostEqual(
            episode.mean_score,
            sum(match.score for match in episode.matches) / 6,
        )
        self.assertEqual(set(episode.scores_by_opponent()), {"all_c", "all_d", "tft"})

    def test_rejects_bad_pool(self):
        with self.assertRaises(ValueError):
            TrainingEnv([])
        with self.assertRaises(ValueError):
            TrainingEnv([make_agent("tft"), make_agent("tft")])
        with self.assertRaises(ValueError):
            TrainingEnv([make_agent("tft")], opponents_per_round=0)


if __name__ == "__main__":
    unittest.main()
