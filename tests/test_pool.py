import random
import unittest

from ipd.pool import FAMILIES, canonical, make_agent, sample_population, uniquify
from ipd.sim import Agent, rules

M20 = (rules.MISSING_ROW,) * 20


def play(agent, opp_moves):
    """让 agent 对一个固定动作序列出手，返回 agent 的动作序列。"""
    my = []
    for length in range(len(opp_moves) + 1):
        if length == rules.TURNS:
            break
        my.append(agent.fn(0, length, 1, tuple(my), 2, tuple(opp_moves[:length]), rules.REPAY, M20, M20))
    return "".join(my)


class LegalityTest(unittest.TestCase):
    def test_sampled_agents_return_legal_moves(self):
        rng = random.Random(0)
        agents = [a for a in sample_population(rng, 300) if "faulty" not in a.name]
        for agent in agents + canonical():
            for _ in range(3):
                opp = "".join(rng.choice("cb") for _ in range(24))
                with self.subTest(agent=agent.name):
                    moves = play(agent, opp)
                    self.assertEqual(len(moves), 25)
                    self.assertTrue(set(moves) <= {"c", "b"})

    def test_every_family_is_sampled(self):
        names = {a.family for a in sample_population(random.Random(0), 2000)}
        self.assertEqual(names, {f.name for f in FAMILIES})

    def test_population_names_unique(self):
        agents = sample_population(random.Random(1), 200)
        self.assertEqual(len({a.name for a in agents}), len(agents))

    def test_uniquify_does_not_stack_suffixes(self):
        tft = make_agent("tft")
        names = [a.name for a in uniquify([tft, tft, Agent("tft#2", tft.fn), tft])]
        self.assertEqual(names, ["tft", "tft#2", "tft#3", "tft#4"])


class BehaviourTest(unittest.TestCase):
    def test_tft_family(self):
        opp = "cbbcc" + "c" * 19
        self.assertEqual(play(make_agent("tft"), opp)[:6], "ccbbcc")
        self.assertEqual(play(make_agent("stft"), opp)[:6], "bcbbcc")
        self.assertEqual(play(make_agent("tf2t"), opp)[:6], "cccbcc")

    def test_grim_and_wsls(self):
        opp = "cbcccc" + "c" * 18
        self.assertEqual(play(make_agent("grim"), opp)[:7], "ccbbbbb")
        # WSLS：(c,b) 输 -> 换成 b；(b,c) 赢 -> 保持 b；(b,c) -> b ...
        self.assertEqual(play(make_agent("wsls"), opp)[:4], "ccbb")

    def test_gradual(self):
        # 对手第 1 手背叛 -> 背叛 1 次 + 合作 2 次；第 5 手再背叛 -> 背叛 2 次
        opp = "bcccbccccc" + "c" * 14
        self.assertEqual(play(make_agent("gradual"), opp)[:10], "cbcccbbccc")

    def test_prober(self):
        self.assertEqual(play(make_agent("prober"), "c" * 24)[:6], "bccbbb")
        # 对手第 3 手背叛，不可剥削 -> 转 TFT
        self.assertEqual(play(make_agent("prober"), "ccbccc" + "c" * 18)[:6], "bccbcc")

    def test_end_defect_and_opening(self):
        tft = make_agent("tft")
        moves = play(make_agent("end_defect", {"k": 3}, [tft]), "c" * 24)
        self.assertEqual(moves, "c" * 22 + "bbb")
        moves = play(make_agent("opening", {"pattern": "bc"}, [tft]), "c" * 24)
        self.assertEqual(moves[:3], "bcc")

    def test_by_phase(self):
        agent = make_agent("by_phase", bases=[make_agent("all_c"), make_agent("all_d"), make_agent("all_c")])
        self.assertEqual(play(agent, "c" * 24), "c" * 10 + "b" * 10 + "c" * 5)


if __name__ == "__main__":
    unittest.main()
