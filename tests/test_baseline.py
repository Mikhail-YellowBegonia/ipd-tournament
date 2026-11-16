"""手工基线（issue #3）：合法性、行为、降级、无状态与性能测试。"""

import copy
import random
import time
import unittest

from ipd.agents import baseline_agent, strategy
from ipd.pool import canonical, make_agent
from ipd.sim import TournamentConfig, rules, run_tournament

C, D, M = rules.COOPERATE, rules.DEFECT, rules.MISSING
M20 = (rules.MISSING_ROW,) * rules.HISTORY_MATCHES
TURNS = rules.TURNS


def call(length, my=(), opp=(), history=M20, opp_history=M20, repay=rules.REPAY, rs=0):
    """按考试九参数接口调用基线。"""
    return strategy(rs, length, 1, tuple(my), 2, tuple(opp), repay, history, opp_history)


def uniform_history(move, matched_move=None):
    """对手历史：20 场、每场 25 手都是 move；其对手的动作默认相同。"""
    row = tuple(move for _ in range(TURNS))
    matched = row if matched_move is None else tuple(matched_move for _ in range(TURNS))
    return (row,) * rules.HISTORY_MATCHES, (matched,) * rules.HISTORY_MATCHES


def random_history(rng, hide_prob=0.0):
    """随机对手历史，可选逐步隐藏。"""
    rows = []
    for _ in range(rules.HISTORY_MATCHES):
        row = [rng.choice("cb") for _ in range(TURNS)]
        if hide_prob:
            row = [M if rng.random() < hide_prob else move for move in row]
        rows.append(tuple(row))
    return tuple(rows), tuple(rows)


def play_direct(opp_name, rounds=1, hide_prob=0.0):
    """基线与一个标准对手直接对局（无历史或按 hide_prob 隐藏）。"""
    cfg = TournamentConfig(rounds=rounds, hide_prob=hide_prob, shuffle_schedule=False)
    return run_tournament(
        [baseline_agent(), make_agent(opp_name)], seed=0, cfg=cfg, keep_matches=True
    )


class LegalityTest(unittest.TestCase):
    def test_returns_legal_move_for_every_length(self):
        rng = random.Random(0)
        history, _ = random_history(rng, hide_prob=0.2)
        for length in range(TURNS):
            my = tuple(rng.choice("cb") for _ in range(length))
            opp = tuple(rng.choice("cb") for _ in range(length))
            with self.subTest(length=length):
                self.assertIn(call(length, my, opp, history, history), (C, D))

    def test_returns_legal_move_with_empty_history(self):
        for length in (0, 1, 12, 24):
            my = (C,) * length
            opp = (D,) * length
            with self.subTest(length=length):
                self.assertIn(call(length, my, opp), (C, D))

    def test_full_hidden_history_equals_empty_history(self):
        rng = random.Random(1)
        my = tuple(rng.choice("cb") for _ in range(15))
        opp = tuple(rng.choice("cb") for _ in range(15))
        hidden = (rules.MISSING_ROW,) * rules.HISTORY_MATCHES
        self.assertEqual(call(15, my, opp, hidden, hidden), call(15, my, opp))

    def test_step_masked_history_does_not_break(self):
        rng = random.Random(2)
        history, opp_history = random_history(rng, hide_prob=0.3)
        for length in range(TURNS):
            with self.subTest(length=length):
                self.assertIn(call(length, (C,) * length, (D,) * length, history, opp_history), (C, D))


class DeterminismTest(unittest.TestCase):
    def test_same_inputs_same_move(self):
        rng = random.Random(3)
        history, _ = random_history(rng, hide_prob=0.1)
        for length in range(TURNS):
            my = tuple(rng.choice("cb") for _ in range(length))
            opp = tuple(rng.choice("cb") for _ in range(length))
            with self.subTest(length=length):
                self.assertEqual(call(length, my, opp, history, history), call(length, my, opp, history, history))

    def test_no_cross_call_state(self):
        rng = random.Random(4)
        hist_a, _ = uniform_history(C)
        hist_b, _ = random_history(rng)
        my_a, opp_a = (C,) * 10, (C,) * 10
        my_b = tuple(rng.choice("cb") for _ in range(20))
        opp_b = tuple(rng.choice("cb") for _ in range(20))
        isolated_a = call(10, my_a, opp_a, hist_a, hist_a)
        isolated_b = call(20, my_b, opp_b, hist_b, hist_b)
        # 交错调用不应互相影响
        self.assertEqual(call(10, my_a, opp_a, hist_a, hist_a), isolated_a)
        self.assertEqual(call(20, my_b, opp_b, hist_b, hist_b), isolated_b)
        self.assertEqual(call(10, my_a, opp_a, hist_a, hist_a), isolated_a)

    def test_inputs_not_mutated(self):
        rng = random.Random(5)
        history, opp_history = random_history(rng, hide_prob=0.1)
        my = tuple(rng.choice("cb") for _ in range(24))
        opp = tuple(rng.choice("cb") for _ in range(24))
        repay = copy.deepcopy(rules.REPAY)
        before = copy.deepcopy((my, opp, history, opp_history, repay))
        call(24, my, opp, history, opp_history, repay)
        after = (my, opp, history, opp_history, repay)
        self.assertEqual(before, after)


class FallbackTest(unittest.TestCase):
    def test_bad_repay_falls_back_to_cooperate(self):
        self.assertEqual(call(10, (C,) * 10, (C,) * 10, repay=None), C)
        self.assertEqual(call(10, (C,) * 10, (C,) * 10, repay=()), C)

    def test_bad_history_falls_back_to_cooperate(self):
        self.assertEqual(call(10, (C,) * 10, (C,) * 10, history=None, opp_history=None), C)
        self.assertEqual(call(10, (C,) * 10, (C,) * 10, history=(None,), opp_history=(None,)), C)

    def test_out_of_range_length_falls_back_to_cooperate(self):
        self.assertEqual(call(TURNS + 5, (C,) * 10, (C,) * 10), C)


class OpeningTest(unittest.TestCase):
    def test_nice_against_unknown_opponent(self):
        self.assertEqual(call(0), C)

    def test_defends_against_habitual_defector(self):
        history, opp_history = uniform_history(D)
        self.assertEqual(call(0, history=history, opp_history=opp_history), D)

    def test_cooperates_against_habitual_cooperator(self):
        history, opp_history = uniform_history(C)
        self.assertEqual(call(0, history=history, opp_history=opp_history), C)

    def test_opening_uses_history_even_when_masked(self):
        history, opp_history = uniform_history(D)
        # 只保留一场可见记录：仍应识别出惯犯
        visible = (rules.MISSING_ROW,) * (rules.HISTORY_MATCHES - 1) + history[:1]
        visible_opp = (rules.MISSING_ROW,) * (rules.HISTORY_MATCHES - 1) + opp_history[:1]
        self.assertEqual(call(0, history=visible, opp_history=visible_opp), D)


class HistoryReadingTest(unittest.TestCase):
    def test_reads_reciprocity_from_history(self):
        """历史显示对手模仿上一手（TFT 式）时，我方刚合作过就继续合作。"""
        facing = tuple("cbcb" * 6 + "c")[:TURNS]
        own = (C,) + facing[:-1]
        history = (own,) * rules.HISTORY_MATCHES
        opp_history = (facing,) * rules.HISTORY_MATCHES
        self.assertEqual(call(1, (C,), (C,), history, opp_history), C)

    def test_masked_cells_are_skipped(self):
        rng = random.Random(6)
        history, opp_history = random_history(rng, hide_prob=0.5)
        # 不应抛异常，且返回合法动作
        self.assertIn(call(12, (C,) * 12, (D,) * 12, history, opp_history), (C, D))


class EndgameDPTest(unittest.TestCase):
    def test_defects_on_last_turn(self):
        history, opp_history = uniform_history(C)
        my = (C,) * (TURNS - 1)
        self.assertEqual(call(TURNS - 1, my, my, history, opp_history), D)

    def test_does_not_defect_before_last_turn(self):
        history, opp_history = uniform_history(C)
        my = (C,) * (TURNS - 2)
        self.assertEqual(call(TURNS - 2, my, my, history, opp_history), C)

    def test_endgame_respects_repay_table(self):
        """收益表改变（合作收益更高）时，末手不再背叛。"""
        history, opp_history = uniform_history(C)
        my = (C,) * (TURNS - 1)
        no_temptation = tuple((4, -6, 6, -1) for _ in range(TURNS))
        self.assertEqual(
            call(TURNS - 1, my, my, history, opp_history, repay=no_temptation), C
        )


class BehaviourTest(unittest.TestCase):
    def test_does_not_get_fooled_by_all_d(self):
        result = play_direct("all_d")
        mine, theirs = result.matches[0].moves_a, result.matches[0].moves_b
        self.assertLessEqual(mine.count(C), 6)
        self.assertTrue(all(move == D for move in mine[-10:]))
        fooled = sum(1 for me, opp in zip(mine, theirs) if me == C and opp == D)
        # 首手善意 + 阶段切换时的一次试探，之后不再送分
        self.assertLessEqual(fooled, 5)
        self.assertEqual(result.faults, [0, 0])

    def test_keeps_cooperating_with_nice_opponents(self):
        for opp in ("all_c", "tft", "tf2t", "grim", "gradual"):
            with self.subTest(opp=opp):
                result = play_direct(opp)
                mine, theirs = result.matches[0].moves_a, result.matches[0].moves_b
                mutual = sum(1 for me, opp_move in zip(mine, theirs) if me == C and opp_move == C)
                self.assertGreaterEqual(mutual, 20)
                self.assertGreaterEqual(result.scores[0], 60)

    def test_no_faults_in_canonical_league(self):
        agents = [baseline_agent()] + canonical()
        result = run_tournament(
            agents, seed=0, cfg=TournamentConfig(rounds=1, hide_prob=0.0)
        )
        self.assertEqual(result.faults, [0] * len(agents))


class PerformanceTest(unittest.TestCase):
    def test_worst_case_decision_under_10ms(self):
        rng = random.Random(7)
        full, _ = random_history(rng)
        masked, _ = random_history(rng, hide_prob=0.1)
        cases = [
            (TURNS - 1, full[-1][: TURNS - 1], full[-1][1:], full, masked),
            (TURNS - 2, masked[-1][: TURNS - 2], masked[-1][2:], masked, full),
            (0, (), (), M20, M20),
        ]
        for case in cases:  # 预热
            call(case[0], case[1], case[2], case[3], case[4])
        times = []
        for _ in range(100):
            for length, my, opp, history, opp_history in cases:
                start = time.perf_counter()
                call(length, my, opp, history, opp_history)
                times.append(time.perf_counter() - start)
        worst = max(times)
        self.assertLess(worst, 0.010, f"worst decision {worst * 1e3:.2f} ms exceeds 10 ms")


class WrapperTest(unittest.TestCase):
    def test_baseline_agent_wrapper(self):
        agent = baseline_agent()
        self.assertEqual(agent.family, "baseline")
        self.assertIs(agent.fn, strategy)


if __name__ == "__main__":
    unittest.main()
