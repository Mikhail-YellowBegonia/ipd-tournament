"""行为指纹与聚类（issue #2）的单元测试。"""

import unittest

from ipd.eval.fingerprint import (
    PROBES,
    classify_fingerprint,
    cluster_fingerprints,
    cluster_weights,
    dedupe,
    distance,
    fingerprint,
)
from ipd.pool import make_agent
from ipd.sim import Agent, rules

C, D = rules.COOPERATE, rules.DEFECT
TURNS = rules.TURNS


def broken(*args):
    raise RuntimeError("boom")


def illegal(*args):
    return None


class FingerprintTest(unittest.TestCase):
    def test_vector_length_and_range(self):
        result = fingerprint(make_agent("tft"))
        self.assertEqual(len(result.vector), len(PROBES) * TURNS)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in result.vector))
        self.assertEqual(len(result.vector), 700)

    def test_deterministic_agent_is_reproducible(self):
        agent = make_agent("tft")
        first = fingerprint(agent)
        second = fingerprint(agent)
        self.assertEqual(first.vector, second.vector)
        self.assertEqual(first.faults, 0)
        self.assertEqual(first.cooperation("all_c@empty"), 1.0)
        self.assertEqual(first.cooperation("all_d@empty"), 1.0 / TURNS)

    def test_stochastic_agent_uses_seed_average(self):
        agent = make_agent("rand", {"p": 0.5})
        first = fingerprint(agent, seeds=(0, 1, 2))
        second = fingerprint(agent, seeds=(0, 1, 2))
        self.assertEqual(first.vector, second.vector)
        self.assertEqual(first.faults, 0)
        self.assertTrue(any(0.0 < value < 1.0 for value in first.vector))
        # 每个探针的 25 手平均应接近 0.5
        average = first.cooperation("all_c@empty")
        self.assertGreater(average, 0.3)
        self.assertLess(average, 0.7)

    def test_stochastic_agent_is_seeded_per_probe(self):
        """每个探针重置全局随机，探针顺序不影响同一探针的结果。"""
        agent = make_agent("rand", {"p": 0.5})
        full = fingerprint(agent)
        only_tft = fingerprint(agent, probes=(PROBES[8],))  # tft@empty
        self.assertEqual(full.vector[8 * TURNS:(8 + 1) * TURNS], only_tft.vector)

    def test_exception_and_illegal_move_become_cooperation(self):
        for fn in (broken, illegal):
            with self.subTest(fn=fn.__name__):
                agent = Agent("bad", fn, family="test", layer="TEST")
                result = fingerprint(agent, seeds=(0,))
                self.assertTrue(all(value == 1.0 for value in result.vector))
                self.assertEqual(result.faults, len(PROBES) * TURNS)

    def test_history_scenarios_change_reputation_behaviour(self):
        """读历史的 reputation_gate 对被剥削历史与互害历史反应不同。"""
        agent = make_agent(
            "reputation_gate", {"threshold": 0.25}, [make_agent("tft")]
        )
        result = fingerprint(agent)
        # all_d@coop：历史主体一直背叛、其对手一直合作 -> 坏信誉，首手即背叛
        self.assertEqual(result.cooperation("all_d@coop"), 0.0)
        # all_d@defect：双方都背叛 -> 不构成无端背叛，仍按 TFT 回应
        self.assertGreater(result.cooperation("all_d@defect"), 0.0)

    def test_empty_seeds_rejected(self):
        with self.assertRaises(ValueError):
            fingerprint(make_agent("tft"), seeds=())


class DistanceAndClusterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agents = [
            make_agent("all_c"),
            make_agent("all_d"),
            make_agent("tft"),
            make_agent("gtft", {"g": 0.05}),
            make_agent("noisy", {"eps": 0.05}, [make_agent("tft")]),
            make_agent("end_defect", {"k": 1}, [make_agent("tft")]),
        ]
        cls.fps = [fingerprint(agent) for agent in cls.agents]

    def test_distance(self):
        self.assertEqual(distance(self.fps[0], self.fps[0]), 0.0)
        self.assertAlmostEqual(distance(self.fps[0], self.fps[1]), 1.0)
        self.assertAlmostEqual(
            distance(self.fps[0], self.fps[1]), distance(self.fps[1], self.fps[0])
        )

    def test_similar_tft_variants_cluster_together(self):
        clusters = cluster_fingerprints(self.fps, threshold=0.12)
        by_name = {member.name: cluster.label for cluster in clusters for member in cluster.members}
        self.assertEqual(by_name["tft"], by_name["gtft(g=0.05)"])
        self.assertEqual(by_name["tft"], by_name["noisy(eps=0.05)[tft]"])
        self.assertEqual(by_name["tft"], by_name["end_defect(k=1)[tft]"])
        self.assertNotEqual(by_name["all_c"], by_name["all_d"])
        self.assertNotEqual(by_name["all_c"], by_name["tft"])

    def test_clusters_are_deterministic_and_have_medoids(self):
        first = cluster_fingerprints(self.fps, threshold=0.12)
        second = cluster_fingerprints(self.fps, threshold=0.12)
        self.assertEqual(
            [(c.label, c.names) for c in first],
            [(c.label, c.names) for c in second],
        )
        for cluster in first:
            self.assertIn(cluster.representative.name, cluster.names)
            # medoid 到簇内其他成员的总距离最小
            total = sum(distance(cluster.representative, member) for member in cluster.members)
            for member in cluster.members:
                self.assertLessEqual(
                    total,
                    sum(distance(member, other) for other in cluster.members) + 1e-12,
                )

    def test_dedupe_and_weights(self):
        clusters = cluster_fingerprints(self.fps, threshold=0.12)
        reps = dedupe(self.fps, threshold=0.12)
        self.assertEqual(len(reps), len(clusters))
        weights = cluster_weights(clusters)
        self.assertEqual(set(weights), {fp.name for fp in self.fps})
        self.assertAlmostEqual(sum(weights.values()), len(clusters))
        for cluster in clusters:
            self.assertAlmostEqual(sum(weights[name] for name in cluster.names), 1.0)

    def test_classify(self):
        by_name = {fp.name: fp for fp in self.fps}
        self.assertEqual(classify_fingerprint(by_name["all_c"]), "cooperative")
        self.assertEqual(classify_fingerprint(by_name["all_d"]), "hostile")
        self.assertEqual(classify_fingerprint(by_name["tft"]), "shrewd")


if __name__ == "__main__":
    unittest.main()
