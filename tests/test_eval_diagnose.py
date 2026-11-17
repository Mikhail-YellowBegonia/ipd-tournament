"""全池诊断 harness（T2）：统计、抽样、tag、尾部、gate、确定性。"""

import json
import unittest

from ipd.eval.diagnose import check_gate, evaluate_diagnose, format_report
from ipd.pool import make_agent
from ipd.sim import Agent, TournamentConfig


def config(rounds=2):
    return TournamentConfig(rounds=rounds, hide_prob=0.0, shuffle_schedule=False)


def custom_pool():
    return [
        Agent(
            name="coop",
            fn=make_agent("all_c").fn,
            params={"tags": ("cooperative",)},
        ),
        Agent(
            name="hostile",
            fn=make_agent("all_d").fn,
            params={"tags": ("hostile",)},
        ),
        Agent(
            name="mirror",
            fn=make_agent("tft").fn,
            params={"tags": ("conditional",)},
        ),
        Agent(
            name="punisher",
            fn=make_agent("grim").fn,
            params={"tags": ("conditional", "history")},
        ),
    ]


class StatisticsTest(unittest.TestCase):
    def test_per_match_uses_full_roster_divisor(self):
        report = evaluate_diagnose(
            make_agent("tft"),
            baseline=make_agent("all_c"),
            clone=False,
            pool=custom_pool(),
            pool_name="custom",
            seeds=(0,),
            cfg=config(rounds=2),
        )
        self.assertEqual(report["n_agents"], 6)  # candidate + baseline + 4
        self.assertEqual(report["size"], 4)
        divisor = 2 * (report["n_agents"] - 1)
        total = report["per_seed"][0]["candidate_total"]
        self.assertAlmostEqual(report["candidate_per_match"], total / divisor)

    def test_solo_per_opponent_sums_to_total(self):
        report = evaluate_diagnose(
            make_agent("tft"),
            baseline=None,
            clone=False,
            pool=custom_pool(),
            pool_name="custom",
            seeds=(0,),
            cfg=config(rounds=2),
        )
        self.assertIsNone(report["gap"])
        self.assertIsNone(report["clone_noise"])
        self.assertFalse(report["clone"])
        total = sum(item["candidate"] for item in report["per_opponent"]) * 2
        self.assertAlmostEqual(total, report["per_seed"][0]["candidate_total"])

    def test_gap_and_noise_are_zero_for_identical_deterministic_strategies(self):
        report = evaluate_diagnose(
            make_agent("all_c"),
            baseline=make_agent("all_c"),
            clone=True,
            pool=custom_pool(),
            pool_name="custom",
            seeds=(0,),
            cfg=config(rounds=2),
        )
        self.assertAlmostEqual(report["gap"], 0.0)
        self.assertAlmostEqual(report["clone_noise"], 0.0)

    def test_delta_matches_candidate_minus_baseline(self):
        report = evaluate_diagnose(
            make_agent("tft"),
            baseline=make_agent("all_c"),
            clone=False,
            pool=custom_pool(),
            pool_name="custom",
            seeds=(0,),
            cfg=config(rounds=2),
        )
        for item in report["per_opponent"]:
            self.assertAlmostEqual(
                item["delta"], item["candidate"] - item["baseline"]
            )


class TagAndTailTest(unittest.TestCase):
    def test_negative_and_worst_delta_tail(self):
        report = evaluate_diagnose(
            make_agent("all_c"),
            baseline=make_agent("all_d"),
            clone=False,
            pool=custom_pool(),
            pool_name="custom",
            seeds=(0,),
            cfg=config(rounds=1),
        )
        negative_names = {item["name"] for item in report["tails"]["negative"]}
        self.assertIn("hostile", negative_names)
        self.assertLessEqual(len(report["tails"]["worst_delta"]), 10)
        self.assertEqual(report["tails"]["worst_delta"][0]["name"], "hostile")

    def test_tag_aggregation(self):
        report = evaluate_diagnose(
            make_agent("all_c"),
            baseline=make_agent("all_d"),
            clone=False,
            pool=custom_pool(),
            pool_name="custom",
            seeds=(0,),
            cfg=config(rounds=1),
        )
        self.assertEqual(report["by_tag"]["cooperative"]["n"], 1)
        self.assertEqual(report["by_tag"]["hostile"]["n"], 1)
        self.assertEqual(report["by_tag"]["conditional"]["n"], 2)
        self.assertEqual(report["by_tag"]["history"]["n"], 1)
        self.assertAlmostEqual(
            report["by_tag"]["history"]["delta"],
            report["by_tag"]["history"]["candidate"]
            - report["by_tag"]["history"]["baseline"],
        )


class SamplingTest(unittest.TestCase):
    def test_size_limits_opponents_and_agents(self):
        pool = [
            make_agent("all_c"),
            make_agent("all_d"),
            make_agent("tft"),
            make_agent("grim"),
            make_agent("tf2t"),
            make_agent("gtft"),
        ]
        report = evaluate_diagnose(
            make_agent("tft"),
            baseline=None,
            clone=False,
            pool=pool,
            pool_name="custom",
            size=3,
            seeds=(0,),
            cfg=config(rounds=1),
        )
        self.assertEqual(report["size"], 3)
        self.assertEqual(len(report["per_opponent"]), 3)
        self.assertEqual(report["n_agents"], 4)  # candidate + 3

    def test_multi_seed_appearances_are_tracked(self):
        pool = [
            make_agent("all_c"),
            make_agent("all_d"),
            make_agent("tft"),
            make_agent("grim"),
            make_agent("tf2t"),
            make_agent("gtft"),
        ]
        report = evaluate_diagnose(
            make_agent("tft"),
            baseline=None,
            clone=False,
            pool=pool,
            pool_name="custom",
            size=3,
            seeds=(0, 1, 2),
            cfg=config(rounds=1),
        )
        for item in report["per_opponent"]:
            self.assertGreaterEqual(item["appearances"], 1)
            self.assertLessEqual(item["appearances"], 3)

    def test_invalid_size_raises(self):
        with self.assertRaises(ValueError):
            evaluate_diagnose(
                make_agent("tft"),
                pool=custom_pool(),
                seeds=(0,),
                size=0,
            )


class DeterminismAndFormatTest(unittest.TestCase):
    def _report(self):
        return evaluate_diagnose(
            make_agent("tft"),
            baseline=make_agent("all_c"),
            clone=True,
            pool=custom_pool(),
            pool_name="custom",
            seeds=(0,),
            cfg=config(rounds=2),
        )

    def test_deterministic_and_json_serializable(self):
        first = self._report()
        second = self._report()
        first.pop("elapsed")
        second.pop("elapsed")
        self.assertEqual(
            json.dumps(first, sort_keys=True, ensure_ascii=False),
            json.dumps(second, sort_keys=True, ensure_ascii=False),
        )

    def test_format_report_contains_sections(self):
        text = format_report(self._report())
        self.assertIn("candidate", text)
        self.assertIn("baseline", text)
        self.assertIn("by tag", text)
        self.assertIn("worst", text)

    def test_variants(self):
        report = evaluate_diagnose(
            make_agent("tft"),
            baseline=make_agent("all_c"),
            clone=False,
            pool=custom_pool(),
            pool_name="custom",
            seeds=(0,),
            cfg=config(rounds=1),
            variants=True,
        )
        self.assertEqual(len(report["variants"]), 4)
        names = {row["name"] for row in report["variants"]}
        self.assertIn("hide_prob=0.0", names)
        self.assertIn("shuffle_schedule=False", names)


class GateTest(unittest.TestCase):
    def test_pass(self):
        passed, reasons = check_gate(
            {"gap": 1.5, "worst_seed_gap": 0.2}, min_gain=1.0, max_worst_drop=1.0
        )
        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_fail_on_mean(self):
        passed, reasons = check_gate(
            {"gap": 0.5, "worst_seed_gap": 0.2}, min_gain=1.0, max_worst_drop=1.0
        )
        self.assertFalse(passed)
        self.assertTrue(any("mean gap" in reason for reason in reasons))

    def test_fail_on_worst_seed(self):
        passed, reasons = check_gate(
            {"gap": 1.5, "worst_seed_gap": -2.0}, min_gain=1.0, max_worst_drop=1.0
        )
        self.assertFalse(passed)
        self.assertTrue(any("worst seed" in reason for reason in reasons))

    def test_fail_without_baseline(self):
        passed, reasons = check_gate({"gap": None, "worst_seed_gap": None})
        self.assertFalse(passed)
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
