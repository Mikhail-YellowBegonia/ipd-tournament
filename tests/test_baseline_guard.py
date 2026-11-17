"""反试探守卫（baseline_guard）：规则、透传、降级与机制测试。"""

import random
import unittest
from unittest import mock

from ipd.agents import baseline
from ipd.agents import baseline_guard
from ipd.sim import Agent, TournamentConfig, rules, run_tournament

C, D, M = rules.COOPERATE, rules.DEFECT, rules.MISSING
M20 = (rules.MISSING_ROW,) * rules.HISTORY_MATCHES
TURNS = rules.TURNS


def call(length, my=(), opp=(), history=M20, opp_history=M20, repay=rules.REPAY, rs=0):
    """按考试九参数接口调用候选。"""
    return baseline_guard.strategy(
        rs, length, 1, tuple(my), 2, tuple(opp), repay, history, opp_history
    )


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


class GuardRuleTest(unittest.TestCase):
    """守卫的触发条件：只在 length==1 且对手首手背叛时改判。"""

    def test_defects_at_second_move_when_opponent_opened_defect(self):
        with mock.patch.object(baseline_guard, "_baseline_strategy", return_value=C):
            self.assertEqual(call(1, my=(C,), opp=(D,)), D)

    def test_passthrough_when_opponent_opened_cooperate(self):
        with mock.patch.object(baseline_guard, "_baseline_strategy", return_value=C):
            self.assertEqual(call(1, my=(C,), opp=(C,)), C)

    def test_passthrough_when_baseline_already_defects(self):
        with mock.patch.object(baseline_guard, "_baseline_strategy", return_value=D):
            self.assertEqual(call(1, my=(C,), opp=(D,)), D)

    def test_guard_only_at_second_move(self):
        for length in (0, 2, 12, 24):
            opp = (D,) + (C,) * (length - 1) if length else ()
            my = (C,) * length
            with self.subTest(length=length):
                with mock.patch.object(
                    baseline_guard, "_baseline_strategy", return_value=C
                ):
                    self.assertEqual(call(length, my=my, opp=opp), C)

    def test_does_not_touch_other_actions(self):
        # 对手首手背叛，但 baseline 在 length!=1 的动作不应被改判。
        with mock.patch.object(baseline_guard, "_baseline_strategy", return_value=C):
            self.assertEqual(call(2, my=(C, C), opp=(D, C)), C)

    def test_always_defects_on_trigger_even_if_baseline_cooperates(self):
        # 真实 baseline：用随机历史，验证触发状态下守卫一定背叛。
        rng = random.Random(0)
        for _ in range(30):
            history, opp_history = random_history(rng, hide_prob=0.2)
            with self.subTest(history=history[:1]):
                self.assertEqual(
                    call(1, my=(C,), opp=(D,), history=history, opp_history=opp_history),
                    D,
                )


class TransparencyTest(unittest.TestCase):
    """非触发状态下必须与 baseline 完全一致。"""

    def test_matches_baseline_when_not_triggered(self):
        rng = random.Random(1)
        for _ in range(50):
            length = rng.randrange(TURNS)
            my = tuple(rng.choice("cb") for _ in range(length))
            opp = tuple(rng.choice("cb") for _ in range(length))
            history, opp_history = random_history(rng, hide_prob=0.1)
            if length == 1 and opp and opp[0] == D:
                continue
            rs = rng.randrange(rules.ROUNDS)
            got = call(length, my, opp, history, opp_history, rs=rs)
            want = baseline.strategy(
                rs, length, 1, my, 2, opp, rules.REPAY, history, opp_history
            )
            with self.subTest(length=length):
                self.assertEqual(got, want)

    def test_opponent_opened_cooperate_matches_baseline_at_second_move(self):
        history, opp_history = uniform_history(C)
        got = call(1, my=(C,), opp=(C,), history=history, opp_history=opp_history)
        want = baseline.strategy(
            0, 1, 1, (C,), 2, (C,), rules.REPAY, history, opp_history
        )
        self.assertEqual(got, want)


class RobustnessTest(unittest.TestCase):
    def test_returns_cooperate_on_exception(self):
        with mock.patch.object(
            baseline_guard, "_baseline_strategy", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(call(1, my=(C,), opp=(D,)), C)

    def test_forwards_all_nine_arguments(self):
        sentinel = object()
        with mock.patch.object(
            baseline_guard, "_baseline_strategy", return_value=sentinel
        ) as patched:
            self.assertIs(
                baseline_guard.strategy(
                    3, 7, 11, ("c",), 22, ("b",), rules.REPAY, M20, M20
                ),
                sentinel,
            )
            patched.assert_called_once_with(
                3, 7, 11, ("c",), 22, ("b",), rules.REPAY, M20, M20
            )

    def test_does_not_mutate_inputs(self):
        my = ("c", "b", "c")
        opp = ("b", "c", "b")
        history, opp_history = uniform_history(C)
        before = (my, opp, history, opp_history)
        call(3, my, opp, history, opp_history)
        self.assertEqual((my, opp, history, opp_history), before)

    def test_returns_legal_move_for_every_length(self):
        rng = random.Random(2)
        for length in range(TURNS):
            my = tuple(rng.choice("cb") for _ in range(length))
            opp = tuple(rng.choice("cb") for _ in range(length))
            history, opp_history = random_history(rng)
            with self.subTest(length=length):
                self.assertIn(call(length, my, opp, history, opp_history), (C, D))

    def test_agent_wrapper(self):
        agent = baseline_guard.baseline_guard_agent("candidate")
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "candidate")
        self.assertIs(agent.fn, baseline_guard.strategy)


def make_prober():
    """留出池 prober 的最小复刻：B-C-C 试探；对手第 2 手仍合作则永久背叛。"""

    def prober(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
        if len(opp_decisions) < 3:
            return ("b", "c", "c")[len(opp_decisions)]
        if opp_decisions[1] == "c":
            return D
        return C if opp_decisions[-1] == "c" else D

    return prober


class ProberMechanismTest(unittest.TestCase):
    """守卫应破掉 prober 的永久背叛触发条件。"""

    def test_guard_prevents_permanent_defection(self):
        cfg = TournamentConfig(rounds=1, hide_prob=0.0, shuffle_schedule=False)
        guard = baseline_guard.baseline_guard_agent()
        prober = Agent(name="prober", fn=make_prober(), family="test", layer="X")
        result = run_tournament([guard, prober], seed=0, cfg=cfg, keep_matches=True)
        self.assertEqual(len(result.matches), 1)
        match = result.matches[0]
        self.assertEqual(match.a, 0)
        self.assertEqual(match.moves_a[1], D)
        # prober 的第 2 手看到的是 D，不会进入永久背叛；后续应有合作。
        self.assertIn(C, match.moves_b[3:])

    def test_baseline_without_guard_triggers_permanent_defection(self):
        cfg = TournamentConfig(rounds=1, hide_prob=0.0, shuffle_schedule=False)
        base = baseline.baseline_agent()
        prober = Agent(name="prober", fn=make_prober(), family="test", layer="X")
        result = run_tournament([base, prober], seed=0, cfg=cfg, keep_matches=True)
        match = result.matches[0]
        self.assertEqual(match.moves_b[3:], (D,) * (TURNS - 3))


if __name__ == "__main__":
    unittest.main()
