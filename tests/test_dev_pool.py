"""Dev 测试池（T3）：契约、独立性、行为覆盖与集成测试。"""

import ast
import inspect
from pathlib import Path
import random
import unittest

from ipd.eval.diagnose import evaluate_diagnose
from ipd.eval.fingerprint import DEFAULT_SEEDS, fingerprint_many
from ipd.pool import agents_for_stage, make_agent
from ipd.pool.dev import (
    DEV_AGENTS,
    EXPECTED_COUNT,
    MODULE_NAMES,
    PARAMETERS,
    STRATEGIES,
    VALID_TAGS,
    agents_with_tag,
    dev_agents,
    module_counts,
    tag_counts,
)
from ipd.sim import TournamentConfig, run_tournament
from ipd.sim import rules

C, D, M = "c", "b", "m"
REPAY = rules.REPAY
EMPTY_ROW = (M,) * rules.TURNS
COOP_ROW = (C,) * rules.TURNS
DEFECT_ROW = (D,) * rules.TURNS


def observation(length=12, my_moves=None, opp_moves=None, history=None, opp_history=None):
    my_moves = tuple(my_moves if my_moves is not None else [C] * length)
    opp_moves = tuple(opp_moves if opp_moves is not None else [C] * length)
    history = history if history is not None else (COOP_ROW,) * 20
    opp_history = opp_history if opp_history is not None else (COOP_ROW,) * 20
    return (3, length, 1, my_moves, 2, opp_moves, REPAY, history, opp_history)


def decisions(fn, obs, seeds=5):
    """同一观测在若干随机种子下的决策集合。"""
    out = []
    for seed in range(seeds):
        random.seed(seed)
        out.append(fn(*obs))
    return out


class DevContractTest(unittest.TestCase):
    def test_pool_size_and_names(self):
        self.assertEqual(len(DEV_AGENTS), EXPECTED_COUNT)
        names = [agent.name for agent in DEV_AGENTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith("dv_") for name in names))

    def test_modules_and_strategies_agree(self):
        self.assertEqual(len(MODULE_NAMES), 6)
        counts = module_counts()
        self.assertEqual(set(counts), set(MODULE_NAMES))
        self.assertTrue(all(count == 10 for count in counts.values()))
        self.assertEqual(sum(counts.values()), EXPECTED_COUNT)
        self.assertEqual(len(STRATEGIES), EXPECTED_COUNT)
        self.assertEqual(
            [agent.name for agent in dev_agents()],
            [name for name, _tags, _fn, _module in STRATEGIES],
        )

    def test_parameters_and_tags(self):
        for name, tags, fn, _module in STRATEGIES:
            with self.subTest(name=name):
                self.assertTrue(name.startswith("dv_"))
                self.assertTrue(tags)
                self.assertTrue(set(tags) <= VALID_TAGS)
                self.assertEqual(list(inspect.signature(fn).parameters), list(PARAMETERS))

    def test_tag_quotas(self):
        total = len(DEV_AGENTS)
        counts = tag_counts()
        self.assertGreaterEqual(counts.get("history", 0) / total, 0.20)
        self.assertGreaterEqual(counts.get("conditional", 0) / total, 0.30)
        self.assertGreaterEqual(counts.get("endgame", 0) / total, 0.10)
        self.assertGreaterEqual(counts.get("noisy", 0) / total, 0.10)

    def test_dev_agents_are_not_in_training_pools(self):
        dev_names = {agent.name for agent in DEV_AGENTS}
        for stage in (1, 2, 3, 4):
            with self.subTest(stage=stage):
                stage_names = {agent.name for agent in agents_for_stage(stage, seed=0)}
                self.assertEqual(dev_names & stage_names, set())

    def test_agents_with_tag(self):
        for tag in ("history", "conditional", "endgame", "noisy"):
            with self.subTest(tag=tag):
                selected = agents_with_tag(tag)
                self.assertEqual(
                    {agent.name for agent in selected},
                    {
                        agent.name
                        for agent in DEV_AGENTS
                        if tag in agent.params["tags"]
                    },
                )
        with self.assertRaises(ValueError):
            agents_with_tag("unknown")


class DevIndependenceTest(unittest.TestCase):
    def test_no_repo_imports_or_forbidden_constructs(self):
        allowed = {"random", "math", "collections", "itertools", "functools"}
        modules = [
            Path(__file__).resolve().parents[1] / "ipd" / "pool" / "dev" / f"{name}.py"
            for name in MODULE_NAMES
        ]
        for path in modules:
            with self.subTest(module=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertIn(alias.name.split(".")[0], allowed)
                    elif isinstance(node, ast.ImportFrom):
                        self.assertEqual(node.level, 0)
                        self.assertIn((node.module or "").split(".")[0], allowed)
                    elif isinstance(node, (ast.Global, ast.Nonlocal)):
                        self.fail(f"{path.name} uses global/nonlocal")
                    elif isinstance(node, ast.Call):
                        func = node.func
                        if isinstance(func, ast.Name) and func.id in {"open", "print", "exec", "eval"}:
                            self.fail(f"{path.name} calls {func.id}")
                        if isinstance(func, ast.Attribute) and func.attr == "seed":
                            if isinstance(func.value, ast.Name) and func.value.id == "random":
                                self.fail(f"{path.name} calls random.seed")

    def test_all_strategies_survive_edge_observations(self):
        cases = (
            observation(
                length=0,
                my_moves=(),
                opp_moves=(),
                history=(EMPTY_ROW,) * 20,
                opp_history=(EMPTY_ROW,) * 20,
            ),
            observation(length=0, my_moves=(), opp_moves=()),
            observation(length=24),
            observation(length=12, opp_moves=[D] * 12),
            observation(
                length=12,
                history=tuple(EMPTY_ROW if i % 2 else COOP_ROW for i in range(20)),
                opp_history=tuple(DEFECT_ROW if i % 3 else EMPTY_ROW for i in range(20)),
            ),
        )
        for name, _tags, fn, _module in STRATEGIES:
            for index, obs in enumerate(cases):
                with self.subTest(name=name, case=index):
                    random.seed(0)
                    try:
                        move = fn(*obs)
                    except Exception as exc:  # noqa: BLE001 - 契约要求不抛异常
                        self.fail(f"{name} raised {type(exc).__name__}: {exc}")
                    self.assertIn(move, (C, D))


class DevBehaviorTest(unittest.TestCase):
    @staticmethod
    def _history_variants():
        mixed = tuple(
            COOP_ROW if row % 2 == 0 else DEFECT_ROW for row in range(20)
        )
        return (
            ((COOP_ROW,) * 20, (COOP_ROW,) * 20),
            ((DEFECT_ROW,) * 20, (DEFECT_ROW,) * 20),
            (mixed, mixed),
            ((DEFECT_ROW,) * 20, (COOP_ROW,) * 20),
            ((COOP_ROW,) * 20, (DEFECT_ROW,) * 20),
        )

    def test_history_tag_really_reads_history(self):
        contexts = (
            (7, (C, D, C, D, C, D, C), (C, C, D, D, C, C, D)),
            (12, (C,) * 12, (C,) * 12),
            (12, (C,) * 12, (D,) * 12),
            (18, (C, D) * 9, (D, C) * 9),
            (3, (C,), (D,)),
        )
        variants = self._history_variants()
        for agent in agents_with_tag("history"):
            with self.subTest(name=agent.name):
                differs = False
                for length, my_moves, opp_moves in contexts:
                    outputs = {
                        tuple(
                            decisions(
                                agent.fn,
                                observation(
                                    length=length,
                                    my_moves=my_moves,
                                    opp_moves=opp_moves,
                                    history=history,
                                    opp_history=opp_history,
                                ),
                            )
                        )
                        for history, opp_history in variants
                    }
                    if len(outputs) > 1:
                        differs = True
                        break
                self.assertTrue(differs, f"{agent.name} never reacts to history")

    def test_conditional_tag_responds_to_current_match(self):
        rng = random.Random(20261117)
        current_cases = []
        for _ in range(24):
            length = rng.randint(0, 24)
            my_moves = tuple(rng.choice((C, D)) for _ in range(length))
            opp_moves = tuple(rng.choice((C, D)) for _ in range(length))
            current_cases.append((length, my_moves, opp_moves))
        current_cases += [
            (12, (C,) * 12, (C,) * 9 + (D,) * 3),
            (12, (C,) * 12, (D,) * 9 + (C,) * 3),
            (12, (C,) * 12, (C,) * 3 + (D,) * 9),
            (12, (C,) * 12, (D,) * 3 + (C,) * 9),
            (18, (C, D) * 9, (C, D) * 9),
            (18, (C, D) * 9, (D, C) * 9),
        ]
        histories = []
        for _ in range(8):
            rows = []
            for _row in range(20):
                if rng.random() < 0.25:
                    rows.append(EMPTY_ROW)
                else:
                    rows.append(tuple(rng.choice((C, D)) for _ in range(rules.TURNS)))
            histories.append(tuple(rows))
        histories += [
            (COOP_ROW,) * 20,
            (DEFECT_ROW,) * 20,
            tuple(COOP_ROW if i % 2 else DEFECT_ROW for i in range(20)),
        ]
        for agent in agents_with_tag("conditional"):
            with self.subTest(name=agent.name):
                reacts = False
                for history in histories:
                    outputs = set()
                    for length, my_moves, opp_moves in current_cases:
                        random.seed(0)
                        outputs.add(
                            agent.fn(*observation(
                                length=length,
                                my_moves=my_moves,
                                opp_moves=opp_moves,
                                history=history,
                                opp_history=history,
                            ))
                        )
                    if len(outputs) > 1:
                        reacts = True
                        break
                self.assertTrue(
                    reacts, f"{agent.name} never reacts to current match"
                )

    def test_endgame_tag_defects_at_final_turn(self):
        obs = observation(length=24, my_moves=[C] * 24, opp_moves=[C] * 24)
        for agent in agents_with_tag("endgame"):
            with self.subTest(name=agent.name):
                random.seed(999)
                self.assertEqual(agent.fn(*obs), D)

    def test_noisy_tag_has_both_actions(self):
        mixed_history = tuple(
            COOP_ROW if row % 2 == 0 else DEFECT_ROW for row in range(20)
        )
        obs = observation(
            length=7,
            my_moves=(C, D, C, D, C, D, C),
            opp_moves=(C, C, D, D, C, C, D),
            history=(COOP_ROW,) * 20,
            opp_history=mixed_history,
        )
        for agent in agents_with_tag("noisy"):
            with self.subTest(name=agent.name):
                random.seed(12345)
                moves = {agent.fn(*obs) for _ in range(60)}
                self.assertEqual(moves, {C, D})

    def test_fingerprint_diversity(self):
        fingerprints = fingerprint_many(DEV_AGENTS, seeds=DEFAULT_SEEDS)
        unique = {
            tuple(round(value, 4) for value in fingerprint.vector)
            for fingerprint in fingerprints
        }
        self.assertGreaterEqual(len(unique), 54)


class DevIntegrationTest(unittest.TestCase):
    def test_full_pool_runs_without_faults(self):
        from ipd.pool import canonical

        agents = list(DEV_AGENTS) + canonical()
        unique = []
        seen = set()
        for agent in agents:
            if agent.name in seen:
                continue
            seen.add(agent.name)
            unique.append(agent)
        cfg = TournamentConfig(rounds=1, hide_prob=0.1, timeout=0.5)
        result = run_tournament(unique, seed=0, cfg=cfg)
        self.assertEqual(sum(result.faults), 0)
        self.assertTrue(all(isinstance(score, int) for score in result.scores))

    def test_diagnose_entry(self):
        cfg = TournamentConfig(rounds=1, hide_prob=0.0, timeout=0.5)
        result = evaluate_diagnose(
            make_agent("tft"),
            baseline=make_agent("all_c"),
            clone=False,
            pool=None,
            pool_name="dev",
            size=6,
            seeds=(0,),
            cfg=cfg,
        )
        self.assertEqual(result["faults"], 0)
        self.assertEqual(result["size"], 6)
        self.assertEqual(result["n_agents"], 8)
        self.assertIsNotNone(result["baseline_per_match"])
        self.assertIsNotNone(result["gap"])

    def test_sampling_is_deterministic(self):
        cfg = TournamentConfig(rounds=1, hide_prob=0.0)
        first = evaluate_diagnose(
            make_agent("tft"), pool=None, pool_name="dev", size=5,
            seeds=(3,), cfg=cfg,
        )
        second = evaluate_diagnose(
            make_agent("tft"), pool=None, pool_name="dev", size=5,
            seeds=(3,), cfg=cfg,
        )
        self.assertEqual(
            [item["name"] for item in first["per_opponent"]],
            [item["name"] for item in second["per_opponent"]],
        )


if __name__ == "__main__":
    unittest.main()
