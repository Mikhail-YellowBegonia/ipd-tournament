import random
import unittest

try:
    import axelrod as axl
except ImportError:
    axl = None

from ipd.pool import make_agent
from ipd.sim import TournamentConfig, run_tournament

if axl is not None:
    from ipd.pool.axl import GAME, axelrod_agents, excluded_strategies

PATTERN = "CCDCDDCCCD"


def to_ours(actions):
    return "".join("c" if a == axl.Action.C else "b" for a in actions)


def play_against(agent, opp, rs=0):
    """让适配后的策略对一个固定动作序列出手。"""
    mine = []
    for t in range(25):
        mine.append(agent.fn(rs, t, 1, tuple(mine), 2, tuple(opp[:t]), None, None, None))
    return "".join(mine)


@unittest.skipIf(axl is None, "axelrod not installed")
class AxelrodAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agents = axelrod_agents()
        cls.classes = {s.__name__: s for s in axl.all_strategies}

    def test_exclusions(self):
        excluded = excluded_strategies()
        names = {a.family for a in self.agents}
        self.assertFalse(names & set(excluded))
        self.assertEqual(len(names) + len(excluded), len(axl.all_strategies))
        self.assertEqual(excluded.get("Darwin"), "cheating")
        self.assertEqual(excluded.get("EvolvedAttention"), "long_run_time")

    def test_deterministic_strategies_match_axelrod(self):
        """确定性策略：适配器与 axl.Match 逐手一致（含有内部状态的策略）。"""
        checked = 0
        for agent in self.agents:
            cls = self.classes[agent.family]
            if cls().classifier["stochastic"]:
                continue
            game = axl.DefaultGame if agent.params.get("game") == "default" else GAME
            match = axl.Match([cls(), axl.Cycler(PATTERN)], turns=25, game=game)
            match.play()
            ref = to_ours(a for a, _ in match.result)
            opp = to_ours(b for _, b in match.result)
            with self.subTest(strategy=agent.family):
                self.assertEqual(play_against(agent, opp), ref)
            checked += 1
        self.assertGreater(checked, 100)

    def test_default_game_fallback_only_where_needed(self):
        fallback = {a.family for a in self.agents if a.params.get("game") == "default"}
        self.assertIn("ZDExtortion", fallback)
        self.assertNotIn("TitForTat", fallback)
        self.assertLess(len(fallback), 20)

    def test_replay_matches_incremental(self):
        """交错调用多场（触发重放）与逐场调用结果相同。"""
        opp = "ccbcbbcccb" * 3
        for name in ("Gradual", "EvolvedFSM16", "OmegaTFT"):
            (agent,) = axelrod_agents(names=[name])
            ref = play_against(agent, opp)
            mine = []
            for t in range(25):
                # 每回合之间插入一次对另一场的调用，并故意给出不一致的历史
                agent.fn(0, t, 1, tuple("b" * t), 2, tuple("c" * t), None, None, None)
                mine.append(agent.fn(0, t, 1, tuple(mine), 2, tuple(opp[:t]), None, None, None))
            with self.subTest(strategy=name):
                self.assertEqual("".join(mine), ref)

    def test_stochastic_reproducible_under_global_seed(self):
        stochastic = [a for a in self.agents if self.classes[a.family]().classifier["stochastic"]][:20]
        opp = "ccbcbbcccb" * 3
        for agent in stochastic:
            random.seed(123)
            first = play_against(agent, opp)
            random.seed(123)
            with self.subTest(strategy=agent.family):
                self.assertEqual(play_against(agent, opp), first)

    def test_all_agents_run_without_faults(self):
        """每个策略在我方模拟器中完整打完一届小型锦标赛，没有故障。"""
        field = [make_agent("tft"), make_agent("all_d"), make_agent("rand", {"p": 0.5})]
        cfg = TournamentConfig(rounds=2, timeout=None)
        for agent in self.agents:
            with self.subTest(strategy=agent.family):
                result = run_tournament([agent] + field, seed=0, cfg=cfg)
                self.assertEqual(result.faults[0], 0)


if __name__ == "__main__":
    unittest.main()
