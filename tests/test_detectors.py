import unittest

from ipd.detectors import match_profile, profile
from ipd.pool import make_agent
from ipd.sim import Agent, TournamentConfig, run_tournament
from ipd.sim import rules

M = rules.MISSING_ROW


def row(s):
    return tuple(s)


class ProfileTest(unittest.TestCase):
    def test_empty_history(self):
        p = profile((M,) * 20, (M,) * 20)
        self.assertEqual(p.matches, 0)
        self.assertTrue(all(p.trials(name) == 0 for name in p.counts))
        self.assertEqual(p.rate("coop"), 0.5)
        self.assertEqual(p.endgame_k(), 0)

    def test_endgame_defection_vs_cooperator(self):
        own = row("c" * 22 + "bbb")
        p = match_profile(own, row("c" * 25))
        self.assertEqual(p.counts["endgame"], [1, 1])
        self.assertEqual(p.endgame_k(), 3)
        self.assertEqual(p.counts["last_defect"], [1, 1])
        self.assertEqual(p.counts["opening_defect"], [0, 1])
        # 终局段不计入无端背叛
        self.assertEqual(p.hits("unprovoked"), 0)
        self.assertEqual(p.counts["exploit"], [3, 25])

    def test_retaliation_is_not_endgame(self):
        # TFT 对 AllD：除首手外全背叛，是报复而非终局背叛
        p = match_profile(row("c" + "b" * 24), row("b" * 25))
        self.assertEqual(p.counts["endgame"], [0, 1])
        self.assertEqual(p.counts["retaliate"], [24, 24])
        self.assertEqual(p.counts["grim"], [1, 1])

    def test_all_d_has_no_endgame_trial(self):
        p = match_profile(row("b" * 25), row("c" * 25))
        self.assertEqual(p.trials("endgame"), 0)
        self.assertEqual(p.counts["opening_defect"], [1, 1])
        self.assertEqual(p.counts["exploit"], [25, 25])

    def test_unprovoked_and_copy(self):
        own = row("ccbc" + "c" * 21)
        opp = row("c" * 25)
        p = match_profile(own, opp)
        self.assertEqual(p.hits("unprovoked"), 1)
        self.assertEqual(p.counts["copy"], [23, 24])

    def test_missing_positions_skipped(self):
        own = row("m" * 5 + "c" * 20)
        opp = row("m" * 5 + "c" * 20)
        p = match_profile(own, opp)
        self.assertEqual(p.trials("coop"), 20)
        self.assertEqual(p.trials("opening_defect"), 0)
        self.assertEqual(p.trials("copy"), 19)

    def test_partial_current_match(self):
        p = match_profile(row("cb"), row("cc"))
        self.assertEqual(p.hits("unprovoked"), 1)
        self.assertEqual(p.trials("endgame"), 0)


class L3Test(unittest.TestCase):
    def play_history(self, subject, reader, rounds=4):
        """subject 先和 all_c 打若干轮积累历史，最后一轮的 reader 对 subject 的动作。"""
        log = []

        def spy(*args):
            move = reader.fn(*args)
            if args[4] == 1:
                log.append((args[0], move))
            return move

        agents = [subject, make_agent("all_c"), Agent("reader", spy)]
        cfg = TournamentConfig(rounds=rounds, hide_prob=0.0, shuffle_schedule=False)
        run_tournament(agents, seed=0, cfg=cfg)
        return "".join(m for rs, m in log if rs == rounds - 1)

    def test_endgame_preempt(self):
        tft = make_agent("tft")
        subject = Agent("end3", make_agent("end_defect", {"k": 3}, [tft]).fn)
        reader = make_agent("endgame_preempt", {"margin": 1, "min_rate": 0.5}, [tft])
        moves = self.play_history(subject, reader)
        # 对手终局背叛 3 手，读者提前 1 手：从第 21 手（索引）开始背叛
        self.assertEqual(moves[:21], "c" * 21)
        self.assertEqual(moves[21:], "bbbb")

    def test_reputation_gate_and_exploit(self):
        tft = make_agent("tft")
        gate = make_agent("reputation_gate", {"threshold": 0.25}, [tft])
        self.assertEqual(self.play_history(make_agent("all_d"), gate), "b" * 25)
        self.assertEqual(self.play_history(make_agent("tft"), gate), "c" * 25)
        exploit = make_agent("exploit_soft", {"threshold": 0.2}, [tft])
        # TFT 曾被 reader 剥削后报复，不可剥削
        self.assertEqual(self.play_history(make_agent("tft"), exploit)[0], "c")

    def test_opening_mirror(self):
        tft = make_agent("tft")
        reader = make_agent("opening_mirror", {"threshold": 0.5}, [tft])
        self.assertEqual(self.play_history(make_agent("stft"), reader)[0], "b")
        self.assertEqual(self.play_history(make_agent("tft"), reader)[0], "c")


if __name__ == "__main__":
    unittest.main()
