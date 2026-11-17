"""独立提交版基线（submission/strategy.py）的契约与等价性测试。"""

import ast
import importlib.util
import inspect
from pathlib import Path
import random
import unittest

from ipd.agents.baseline import baseline_agent, strategy as reference_strategy
from ipd.pool.registry import agents_for_stage
from ipd.sim import Agent, TournamentConfig, rules, run_tournament

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "submission" / "strategy.py"
PARAMETERS = (
    "rs",
    "length",
    "my_id",
    "my_decisions",
    "opp_id",
    "opp_decisions",
    "repay",
    "opp_history_decisions",
    "opp_opp_history_decisions",
)


def load_standalone():
    spec = importlib.util.spec_from_file_location(
        "standalone_submission_strategy", STRATEGY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_observation(rng):
    length = rng.randint(0, rules.TURNS - 1)
    rs = rng.randint(0, rules.ROUNDS - 1)
    my = tuple(rng.choice(("c", "b")) for _ in range(length))
    opp = tuple(rng.choice(("c", "b")) for _ in range(length))
    history = tuple(
        tuple(rng.choice(("c", "b", "m")) for _ in range(rules.TURNS))
        for _ in range(rules.HISTORY_MATCHES)
    )
    opp_history = tuple(
        tuple(rng.choice(("c", "b", "m")) for _ in range(rules.TURNS))
        for _ in range(rules.HISTORY_MATCHES)
    )
    return (rs, length, 1, my, 2, opp, rules.REPAY, history, opp_history)


class StandaloneBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_standalone()

    def test_contract(self):
        self.assertTrue(callable(self.module.strategy))
        self.assertEqual(
            list(inspect.signature(self.module.strategy).parameters),
            list(PARAMETERS),
        )
        source = STRATEGY_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertEqual(imports, [], "standalone strategy must not import anything")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                self.fail("standalone strategy must not use global/nonlocal")
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in {"open", "print"}:
                    self.fail(f"standalone strategy must not call {func.id}")
        self.assertNotIn("ipd", source)

    def test_actions_match_reference(self):
        rng = random.Random(0)
        for index in range(500):
            observation = random_observation(rng)
            self.assertEqual(
                self.module.strategy(*observation),
                reference_strategy(*observation),
                f"mismatch at random observation {index}",
            )

    def test_tournament_scores_match_reference(self):
        cfg = TournamentConfig(rounds=2, hide_prob=0.1, timeout=0.5)
        opponents = agents_for_stage(1, seed=0)
        standalone = Agent("standalone", self.module.strategy)
        result = run_tournament([standalone] + opponents, seed=0, cfg=cfg)
        reference = run_tournament(
            [baseline_agent()] + opponents, seed=0, cfg=cfg
        )
        self.assertEqual(result.scores, reference.scores)
        self.assertEqual(sum(result.faults), 0)

    def test_runtime(self):
        history = tuple([("c",) * rules.TURNS] * rules.HISTORY_MATCHES)
        opp_history = tuple([("b",) * rules.TURNS] * rules.HISTORY_MATCHES)
        observation = (
            3,
            24,
            1,
            ("c",) * 24,
            2,
            ("b",) * 24,
            rules.REPAY,
            history,
            opp_history,
        )
        import time

        start = time.perf_counter()
        for _ in range(100):
            self.assertIn(self.module.strategy(*observation), rules.LEGAL_MOVES)
        average = (time.perf_counter() - start) / 100
        self.assertLess(average, 0.01, f"average {average * 1e3:.3f} ms > 10 ms")

    def test_description_template_exists(self):
        path = ROOT / "submission" / "description_template.txt"
        self.assertTrue(path.exists())
        self.assertLessEqual(len(path.read_text(encoding="utf-8")), 600)


if __name__ == "__main__":
    unittest.main()
