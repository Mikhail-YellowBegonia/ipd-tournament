"""评估协议（issue #2）的单元测试。"""

import tempfile
from pathlib import Path
import unittest

from ipd.eval.fingerprint import Cluster, Fingerprint
from ipd.eval.protocol import (
    DedupedPool,
    LAYER_WEIGHTS,
    allocate_counts,
    build_archetype_mixes,
    build_deduped_pool,
    build_layer_mixes,
    evaluate_agent,
    evaluate_mix,
)
from ipd.eval.spec import load_agent
from ipd.pool import make_agent
from ipd.sim import Agent, TournamentConfig


def _fake_pool():
    """构造每簇一个代表的池子，专注测试配比构造，不依赖真实指纹。"""
    agents, fingerprints, clusters, categories = [], [], [], {}
    tft = make_agent("tft")
    labels = ("cooperative", "hostile", "shrewd")
    index = 0
    for layer, count in (("L0", 3), ("L1", 5), ("L2", 5), ("L3", 5)):
        for i in range(count):
            name = f"{layer}-{i}"
            agent = Agent(name, tft.fn, family="tft", layer=layer)
            fp = Fingerprint(name, "tft", layer, (float(index),), seeds=(0,))
            cluster = Cluster(label=f"c{index:02d}", members=[fp], representative=fp)
            agents.append(agent)
            fingerprints.append(fp)
            clusters.append(cluster)
            categories[cluster.label] = labels[index % len(labels)]
            index += 1
    return DedupedPool(
        reservoir=tuple(agents),
        fingerprints=tuple(fingerprints),
        clusters=tuple(clusters),
        categories=categories,
        threshold=0.12,
    )


class AllocationTest(unittest.TestCase):
    def test_allocate_counts(self):
        self.assertEqual(allocate_counts(20, LAYER_WEIGHTS), {"L0": 3, "L1": 9, "L2": 8})
        self.assertEqual(allocate_counts(15, LAYER_WEIGHTS), {"L0": 2, "L1": 7, "L2": 6})
        self.assertEqual(allocate_counts(10, LAYER_WEIGHTS), {"L0": 1, "L1": 5, "L2": 4})
        for total in range(0, 30):
            counts = allocate_counts(total, LAYER_WEIGHTS)
            self.assertEqual(sum(counts.values()), total)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            allocate_counts(-1, LAYER_WEIGHTS)
        with self.assertRaises(ValueError):
            allocate_counts(5, {"L0": 0})


class MixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = _fake_pool()

    def test_layer_mixes_l3_share(self):
        mixes = build_layer_mixes(self.pool, size=6, seed=0)
        self.assertEqual([mix.name for mix in mixes], ["L3=0%", "L3=25%", "L3=50%"])
        expected_l3 = {"L3=0%": 0, "L3=25%": 2, "L3=50%": 3}
        for mix in mixes:
            self.assertEqual(mix.size, 6)
            self.assertEqual(len({agent.name for agent in mix.agents}), 6)
            self.assertEqual(mix.layer_counts()["L3"], expected_l3[mix.name])
            self.assertIn("L0=", mix.description)
            self.assertIn("L3=", mix.description)

    def test_archetype_mixes(self):
        mixes = build_archetype_mixes(self.pool, size=6, seed=0)
        self.assertEqual(
            [mix.name for mix in mixes], ["coop-heavy", "hostile-heavy", "shrewd-heavy"]
        )
        reps_by_category = {}
        for cluster in self.pool.clusters:
            category = self.pool.categories[cluster.label]
            reps_by_category.setdefault(category, set()).add(cluster.representative.name)
        for mix in mixes:
            self.assertEqual(mix.size, 6)
            self.assertEqual(len({agent.name for agent in mix.agents}), 6)
            target = reps_by_category.get(mix.archetype, set())
            selected = {agent.name for agent in mix.agents} & target
            self.assertGreaterEqual(len(selected), min(3, len(target)))
            self.assertIn("L3=", mix.description)

    def test_mixes_are_deterministic(self):
        first = build_layer_mixes(self.pool, size=6, seed=7)
        second = build_layer_mixes(self.pool, size=6, seed=7)
        self.assertEqual(
            [(mix.name, tuple(a.name for a in mix.agents)) for mix in first],
            [(mix.name, tuple(a.name for a in mix.agents)) for mix in second],
        )

    def test_build_deduped_pool_smoke(self):
        pool = build_deduped_pool(pool_seeds=(0,), variants=0)
        self.assertGreater(len(pool.reservoir), 20)
        self.assertGreater(len(pool.clusters), 0)
        self.assertEqual(len(pool.representatives), len(pool.clusters))
        self.assertEqual(set(pool.categories), {c.label for c in pool.clusters})
        self.assertEqual(len(pool.representative_fingerprints), len(pool.clusters))


class EvaluateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = _fake_pool()
        cls.mixes = build_layer_mixes(cls.pool, size=4, seed=0)[:1]

    def test_mix_result_fields(self):
        result = evaluate_mix(
            load_agent("tft"),
            self.mixes[0],
            seeds=(0, 1),
            cfg=TournamentConfig(rounds=1),
        )
        self.assertEqual(result.name, "L3=0%")
        self.assertEqual(result.n_agents, 5)
        self.assertEqual(len(result.seed_scores), 2)
        self.assertGreaterEqual(result.rank, 1)
        self.assertLessEqual(result.rank, result.n_agents)
        self.assertEqual(
            result.per_match,
            result.total / (2 * 1 * self.mixes[0].size),
        )

    def test_protocol_result_summary(self):
        result = evaluate_agent(
            load_agent("tft"),
            self.mixes,
            seeds=(0,),
            cfg=TournamentConfig(rounds=1),
        )
        self.assertEqual(result.agent_name, "tft")
        self.assertIn("agent: tft", result.table())
        self.assertIn("summary:", result.table())
        self.assertIs(result.worst_mix, result.mixes[0])


class AgentSpecTest(unittest.TestCase):
    def test_family(self):
        agent = load_agent("tft")
        self.assertEqual(agent.family, "tft")
        self.assertEqual(agent.layer, "L1")

    def test_family_with_params_and_base(self):
        agent = load_agent("end_defect(k=1)[tft]")
        self.assertEqual(agent.name, "end_defect(k=1)[tft]")
        self.assertEqual(agent.layer, "L2")

    def test_by_phase_with_three_bases(self):
        agent = load_agent("by_phase[tft|tf2t|grim]")
        self.assertEqual(agent.family, "by_phase")
        self.assertEqual(len(agent.params["bases"]), 3)

    def test_unquoted_pattern_value(self):
        agent = load_agent("periodic(pattern=ccb)")
        self.assertEqual(agent.params["pattern"], "ccb")

    def test_module_factory_and_strategy(self):
        factory = load_agent("ipd.agents.baseline:baseline_agent")
        self.assertEqual(factory.family, "baseline")
        strategy = load_agent("ipd.agents.baseline:strategy")
        self.assertEqual(strategy.layer, "CUSTOM")

    def test_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "my_agent.py"
            path.write_text(
                "def strategy(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):\n"
                "    return 'c'\n",
                encoding="utf-8",
            )
            agent = load_agent(str(path))
            self.assertEqual(agent.name, str(path))
            moves = [
                agent.fn(0, length, 1, (), 2, (), None, None, None)
                for length in range(25)
            ]
            self.assertEqual(set(moves), {"c"})

    def test_unknown_family(self):
        with self.assertRaises(ValueError):
            load_agent("no_such_family")


if __name__ == "__main__":
    unittest.main()
