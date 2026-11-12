import unittest

from ipd.pool import make_agent
from ipd.sim import Agent, TournamentConfig, run_tournament
from ipd.sim import rules

NO_HIDE = TournamentConfig(hide_prob=0.0, shuffle_schedule=False)


def one_match(a, b, cfg=None):
    cfg = cfg or TournamentConfig(rounds=1, hide_prob=0.0, shuffle_schedule=False)
    result = run_tournament([a, b], seed=0, cfg=cfg, keep_matches=True)
    return result


class RulesTest(unittest.TestCase):
    def test_repay_shape_and_phases(self):
        self.assertEqual(len(rules.REPAY), 25)
        self.assertEqual(rules.REPAY[0], (4, -6, 2, -1))
        self.assertEqual(rules.REPAY[9], (4, -6, 2, -1))
        self.assertEqual(rules.REPAY[10], (5, -7, 3, -2))
        self.assertEqual(rules.REPAY[20], (6, -8, 4, -3))
        self.assertEqual(rules.REPAY[24], (6, -8, 4, -3))

    def test_payoff(self):
        row = (4, -6, 2, -1)
        self.assertEqual(rules.payoff("c", "c", row), 2)
        self.assertEqual(rules.payoff("c", "b", row), -6)
        self.assertEqual(rules.payoff("b", "c", row), 4)
        self.assertEqual(rules.payoff("b", "b", row), -1)


class MatchScoreTest(unittest.TestCase):
    def test_reference_scores(self):
        cases = [
            ("all_c", "all_c", 70, 70),
            ("all_d", "all_d", -45, -45),
            ("all_d", "all_c", 120, -170),
            # TFT 首手被骗 -6，之后全互相背叛：-9 -20 -15
            ("tft", "all_d", -50, -40),
        ]
        for a, b, sa, sb in cases:
            with self.subTest(a=a, b=b):
                b_agent = make_agent(b)
                if b == a:
                    b_agent = Agent(b_agent.name + "#2", b_agent.fn)
                result = one_match(make_agent(a), b_agent)
                self.assertEqual(result.scores, [sa, sb])


def recorder(log):
    """记录每次调用收到的参数，始终合作。"""
    def s(*args):
        log.append(args)
        return "c"
    return s


class HistoryTest(unittest.TestCase):
    def test_history_shape_padding_and_order(self):
        log = []
        agents = [Agent("rec", recorder(log)), make_agent("all_d"), make_agent("all_c")]
        run_tournament(agents, seed=0, cfg=TournamentConfig(rounds=2, hide_prob=0.0, shuffle_schedule=False))
        # 第 0 轮第一场：对手无历史，全为 "m"
        first = log[0]
        oh, ooh = first[7], first[8]
        self.assertEqual(len(oh), 20)
        self.assertTrue(all(len(row) == 25 for row in oh + ooh))
        self.assertTrue(all(row == rules.MISSING_ROW for row in oh))
        # 第 1 轮第一场对 all_d：all_d 已打过 2 场，最新记录在最后
        call = next(c for c in log if c[0] == 1 and c[4] == 2)
        oh, ooh = call[7], call[8]
        self.assertEqual(oh[:-2], (rules.MISSING_ROW,) * 18)
        self.assertEqual(oh[-1], ("b",) * 25)
        self.assertEqual(ooh[-1], ("c",) * 25)

    def test_call_arguments(self):
        log = []
        one_match(Agent("rec", recorder(log)), make_agent("all_d"))
        self.assertEqual(len(log), 25)
        for length, args in enumerate(log):
            rs, ln, my_id, my, opp_id, opp, repay = args[:7]
            self.assertEqual((rs, ln, my_id, opp_id), (0, length, 1, 2))
            self.assertEqual(my, ("c",) * length)
            self.assertEqual(opp, ("b",) * length)
            self.assertIs(repay, rules.REPAY)

    def test_hidden_records_synced(self):
        log = []
        agents = [Agent("rec", recorder(log)), make_agent("all_d"), make_agent("all_c")]
        for scope in ("match", "step"):
            log.clear()
            cfg = TournamentConfig(rounds=8, hide_prob=0.5, hide_scope=scope)
            run_tournament(agents, seed=1, cfg=cfg)
            for call in log:
                for own, opp in zip(call[7], call[8]):
                    self.assertEqual([x == "m" for x in own], [x == "m" for x in opp])

    def test_round_update_hides_current_round(self):
        log = []
        agents = [make_agent("all_d"), make_agent("all_c"), Agent("rec", recorder(log))]
        cfg = TournamentConfig(rounds=1, hide_prob=0.0, shuffle_schedule=False, history_update="round")
        run_tournament(agents, seed=0, cfg=cfg)
        self.assertTrue(all(row == rules.MISSING_ROW for call in log for row in call[7]))


class FaultTest(unittest.TestCase):
    def test_faults_become_cooperation_and_lock_after_limit(self):
        calls = []

        def bad(*args):
            calls.append(1)
            if len(calls) <= 3:
                raise RuntimeError
            return "b"

        result = one_match(Agent("bad", bad), make_agent("all_c"))
        self.assertEqual(result.faults[0], 3)
        # 3 次故障后不再调用，全场按合作处理
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.matches[0].moves_a, ("c",) * 25)

    def test_illegal_return_counts_as_fault(self):
        result = one_match(Agent("none", lambda *a: None), make_agent("all_c"))
        self.assertEqual(result.faults[0], 3)

    def test_ranking_ties(self):
        result = run_tournament([make_agent("all_c"), make_agent("tft")], seed=0, cfg=NO_HIDE)
        self.assertEqual([r[0] for r in result.ranking()], [1, 1])


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_result(self):
        agents = [make_agent("rand", {"p": 0.5}), make_agent("gtft", {"g": 0.3}), make_agent("tft")]
        a = run_tournament(agents, seed=7)
        b = run_tournament(agents, seed=7)
        self.assertEqual(a.scores, b.scores)


if __name__ == "__main__":
    unittest.main()
