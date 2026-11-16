"""分阶段对手池配置（issue #4）：层组成、数量、可复现性与 TournamentConfig。"""

import random
import unittest

from ipd.pool import (
    STAGE_LAYERS,
    agents_for_stage,
    canonical,
    config_for_stage,
    standard_agent,
)
from ipd.pool.registry import FAMILIES, STAGE_VARIANTS
from ipd.sim import rules

C, D = rules.COOPERATE, rules.DEFECT
M20 = (rules.MISSING_ROW,) * rules.HISTORY_MATCHES
TURNS = rules.TURNS

# 无 seed 时的规模 = 各层所有族的标准版本数。
BASE_COUNTS = {1: 16, 2: 21, 3: 25, 4: 25}
# 给定 seed 时再加上每个可随机族 2 个变体。
SEED_COUNTS = {1: 26, 2: 41, 3: 53, 4: 53}
# 可随机族 = 有参数采样器的族 + by_phase（三个基础策略随机）。
RANDOMIZABLE = {f.name for f in FAMILIES if f.sample is not None} | {"by_phase"}

LAYERS_OF = {f.name: f.layer for f in FAMILIES}


def play(agent, opp_moves):
    """让 agent 对固定动作序列出手，返回其 25 手动作串。"""
    my = []
    for length in range(TURNS):
        my.append(
            agent.fn(0, length, 1, tuple(my), 2, tuple(opp_moves[:length]), rules.REPAY, M20, M20)
        )
    return "".join(my)


class CountTest(unittest.TestCase):
    def test_base_counts(self):
        for stage, expected in BASE_COUNTS.items():
            with self.subTest(stage=stage):
                self.assertEqual(len(agents_for_stage(stage)), expected)

    def test_seed_counts(self):
        for stage, expected in SEED_COUNTS.items():
            with self.subTest(stage=stage):
                self.assertEqual(len(agents_for_stage(stage, seed=0)), expected)

    def test_variants_zero_equals_base(self):
        for stage in STAGE_LAYERS:
            with self.subTest(stage=stage):
                base = [a.name for a in agents_for_stage(stage)]
                self.assertEqual([a.name for a in agents_for_stage(stage, seed=3, variants=0)], base)

    def test_counts_match_formula(self):
        for stage, layers in STAGE_LAYERS.items():
            families = [f for f in FAMILIES if f.layer in layers]
            n_random = sum(f.name in RANDOMIZABLE for f in families)
            with self.subTest(stage=stage):
                self.assertEqual(
                    len(agents_for_stage(stage, seed=1)),
                    len(families) + STAGE_VARIANTS * n_random,
                )


class LayerTest(unittest.TestCase):
    def test_layers_subset_of_stage_layers(self):
        for stage, layers in STAGE_LAYERS.items():
            for agent in agents_for_stage(stage, seed=2):
                with self.subTest(stage=stage, agent=agent.name):
                    self.assertIn(agent.layer, layers)
                    self.assertIn(agent.family, LAYERS_OF)

    def test_layer_boundaries(self):
        self.assertEqual({a.layer for a in agents_for_stage(1)}, {"L0", "L1"})
        self.assertEqual({a.layer for a in agents_for_stage(2)}, {"L0", "L1", "L2"})
        self.assertEqual({a.layer for a in agents_for_stage(3)}, {"L0", "L1", "L2", "L3"})

    def test_all_families_of_layers_present(self):
        for stage, layers in STAGE_LAYERS.items():
            families = {a.family for a in agents_for_stage(stage)}
            with self.subTest(stage=stage):
                self.assertEqual(families, {f.name for f in FAMILIES if f.layer in layers})

    def test_standard_agent_layer_and_family(self):
        for fam in FAMILIES:
            agent = standard_agent(fam.name)
            with self.subTest(family=fam.name):
                self.assertEqual(agent.family, fam.name)
                self.assertEqual(agent.layer, fam.layer)


class UniquenessTest(unittest.TestCase):
    def test_names_unique(self):
        for stage in STAGE_LAYERS:
            for seed in (None, 0, 7):
                with self.subTest(stage=stage, seed=seed):
                    names = [a.name for a in agents_for_stage(stage, seed=seed)]
                    self.assertEqual(len(names), len(set(names)))

    def test_seeded_pools_are_nested(self):
        stage1 = {a.name for a in agents_for_stage(1, seed=5)}
        stage2 = {a.name for a in agents_for_stage(2, seed=5)}
        stage3 = {a.name for a in agents_for_stage(3, seed=5)}
        self.assertLess(stage1, stage2)
        self.assertLess(stage2, stage3)

    def test_stage4_equals_stage3(self):
        for seed in (None, 0):
            with self.subTest(seed=seed):
                self.assertEqual(
                    [a.name for a in agents_for_stage(4, seed=seed)],
                    [a.name for a in agents_for_stage(3, seed=seed)],
                )


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_pool(self):
        for stage in STAGE_LAYERS:
            with self.subTest(stage=stage):
                self.assertEqual(
                    [a.name for a in agents_for_stage(stage, seed=11)],
                    [a.name for a in agents_for_stage(stage, seed=11)],
                )

    def test_seed_changes_variants_only(self):
        base = [a.name for a in agents_for_stage(2)]
        a = [x for x in (ag.name for ag in agents_for_stage(2, seed=1))]
        b = [x for x in (ag.name for ag in agents_for_stage(2, seed=2))]
        self.assertEqual(a[: len(base)], base)
        self.assertEqual(b[: len(base)], base)
        self.assertNotEqual(a, b)


class CallableTest(unittest.TestCase):
    def test_every_agent_plays_legal_moves(self):
        rng = random.Random(0)
        for stage in STAGE_LAYERS:
            for agent in agents_for_stage(stage, seed=0):
                if "faulty" in agent.name:
                    continue  # faulty 族故意抛异常，模拟有 bug 的提交
                opp = [rng.choice("cb") for _ in range(TURNS - 1)]
                with self.subTest(stage=stage, agent=agent.name):
                    moves = play(agent, opp)
                    self.assertEqual(len(moves), TURNS)
                    self.assertLessEqual(set(moves), {C, D})

    def test_pool_runs_in_tournament(self):
        from ipd.sim import TournamentConfig, run_tournament

        agents = agents_for_stage(1, seed=0)
        result = run_tournament(agents, seed=0, cfg=TournamentConfig(rounds=1, hide_prob=0.0))
        self.assertEqual(len(result.scores), len(agents))
        # 阶段 1 不含 faulty 族，不应有故障
        self.assertEqual(result.faults, [0] * len(agents))


class ConfigTest(unittest.TestCase):
    def test_hide_prob_by_stage(self):
        expected = {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.1}
        for stage, prob in expected.items():
            cfg = config_for_stage(stage)
            with self.subTest(stage=stage):
                self.assertEqual(cfg.hide_prob, prob)
                if stage >= 3:
                    self.assertEqual(cfg.hide_scope, "match")

    def test_other_fields_are_defaults(self):
        cfg = config_for_stage(1)
        self.assertEqual(cfg.rounds, rules.ROUNDS)
        self.assertEqual(cfg.history_matches, rules.HISTORY_MATCHES)
        self.assertEqual(config_for_stage(1), config_for_stage(2))
        self.assertEqual(config_for_stage(3), config_for_stage(4))

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            config_for_stage(0)
        with self.assertRaises(ValueError):
            config_for_stage(5)
        with self.assertRaises(ValueError):
            agents_for_stage(5)

    def test_configs_are_usable(self):
        from ipd.sim import run_tournament

        for stage in STAGE_LAYERS:
            with self.subTest(stage=stage):
                agents = agents_for_stage(stage, seed=0)
                result = run_tournament(agents, seed=0, cfg=config_for_stage(stage))
                self.assertEqual(len(result.scores), len(agents))


class CanonicalRegressionTest(unittest.TestCase):
    def test_canonical_unchanged(self):
        names = [a.name for a in canonical()]
        self.assertEqual(len(names), 23)
        self.assertEqual(
            names,
            [
                "all_c",
                "all_d",
                "rand(p=0.5)",
                "periodic(pattern=cb)",
                "tft",
                "stft",
                "tf2t",
                "gtft(g=0.3)",
                "grim",
                "wsls",
                "joss(p=0.1)",
                "gradual",
                "prober",
                "soft_majority",
                "hard_majority",
                "end_defect(k=1)[tft]",
                "end_defect(k=3)[tft]",
                "opening(pattern=b)[grim]",
                "noisy(eps=0.05)[tft]",
                "reputation_gate(threshold=0.25)[tft]",
                "endgame_preempt(margin=1,min_rate=0.5)[tft]",
                "exploit_soft(threshold=0.2)[tft]",
                "opening_mirror(threshold=0.5)[tft]",
            ],
        )


if __name__ == "__main__":
    unittest.main()
