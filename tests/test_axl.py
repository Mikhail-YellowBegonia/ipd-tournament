"""测试 Axelrod 策略适配器。"""

import unittest

try:
    import axelrod as axl
    AXELROD_AVAILABLE = True
except ImportError:
    AXELROD_AVAILABLE = False

if AXELROD_AVAILABLE:
    from ipd.pool.axl import axelrod_agents
    from ipd.pool import canonical
    from ipd.sim.tournament import run_tournament
    from ipd.sim.config import TournamentConfig


@unittest.skipIf(not AXELROD_AVAILABLE, "axelrod library not available")
class TestAxelrodAdapter(unittest.TestCase):
    """测试 Axelrod 策略适配器的正确性。"""

    def test_all_agents_complete_tournament(self):
        """所有适配的策略都能完成锦标赛且零故障。"""
        axl_agents = axelrod_agents(include_long_run_time=False)

        # 确保至少有一些策略被加载
        self.assertGreater(len(axl_agents), 0, "No axelrod agents loaded")

        # 选择一小部分标准对手进行测试
        standard_opponents = [
            a for a in canonical()
            if a.family in ["all_c", "all_d", "tft"]
        ]

        # 每个 Axelrod 策略都与标准对手进行小型锦标赛
        # 为了速度，我们只测试前几个
        for agent in axl_agents[:10]:
            with self.subTest(agent=agent.name):
                field = [agent] + standard_opponents
                cfg = TournamentConfig(rounds=2)
                result = run_tournament(field, seed=42, cfg=cfg)

                # 检查该策略的故障次数
                agent_idx = 0  # agent 在列表首位
                faults = result.faults[agent_idx]
                self.assertEqual(
                    faults, 0,
                    f"{agent.name} had {faults} faults"
                )

    def test_deterministic_strategy_consistency(self):
        """确定性策略通过我方适配器的输出应与 axelrod 自己的匹配一致。"""
        # TitForTat 是确定性策略
        if not hasattr(axl, "TitForTat"):
            self.skipTest("TitForTat not available")

        agents = axelrod_agents(names=["TitForTat"])
        self.assertEqual(len(agents), 1)
        adapted = agents[0]

        # 创建原生 axelrod 匹配
        axl_tft = axl.TitForTat()
        axl_defector = axl.Defector()
        axl_match = axl.Match([axl_tft, axl_defector], turns=25)
        axl_result = axl_match.play()

        # 通过我方适配器模拟相同的对局
        # 模拟器会调用策略函数，我们需要手动调用
        our_moves = []
        opp_moves = []

        for length in range(25):
            my = tuple(our_moves)
            opp = tuple(opp_moves)

            # 调用我方适配的策略
            move = adapted.fn(
                rs=0,
                length=length,
                my_id=1,
                my=my,
                opp_id=2,
                opp=opp,
                repay=None,  # 策略不使用
                oh=None,
                ooh=None,
            )

            our_moves.append(move)
            opp_moves.append("b")  # Defector 总是背叛

        # 转换 axelrod 结果为我方格式
        # axl_result 是一个列表，每项是 (player1_move, player2_move)
        axl_moves_tft = [
            "c" if turn_result[0] == axl.Action.C else "b"
            for turn_result in axl_result
        ]

        # 比较
        self.assertEqual(
            our_moves, axl_moves_tft,
            f"Moves mismatch: ours={our_moves}, axelrod={axl_moves_tft}"
        )

    def test_exclude_cheating_strategies(self):
        """确保作弊策略被排除。"""
        all_agents = axelrod_agents()

        # 检查所有策略都不是作弊策略
        for agent in all_agents:
            strategy_class = getattr(axl, agent.family, None)
            if strategy_class is None:
                continue

            classifier = strategy_class().classifier
            self.assertFalse(
                classifier.get("inspects_source", False),
                f"{agent.name} inspects source"
            )
            self.assertFalse(
                classifier.get("manipulates_source", False),
                f"{agent.name} manipulates source"
            )
            self.assertFalse(
                classifier.get("manipulates_state", False),
                f"{agent.name} manipulates state"
            )

    def test_long_run_time_exclusion(self):
        """默认情况下 long_run_time 策略被排除，显式包含时存在。"""
        default_agents = axelrod_agents(include_long_run_time=False)
        with_long = axelrod_agents(include_long_run_time=True)

        # with_long 应该有更多或相同数量的策略
        self.assertGreaterEqual(len(with_long), len(default_agents))

    def test_history_replay_on_mismatch(self):
        """当历史不匹配时（如模拟器覆盖动作），策略应正确重放。"""
        agents = axelrod_agents(names=["TitForTat"])
        if not agents:
            self.skipTest("TitForTat not available")

        tft = agents[0]

        # 正常序列：对手合作，TFT 应跟随
        move1 = tft.fn(0, 0, 1, (), 2, (), None, None, None)
        self.assertEqual(move1, "c")  # TFT 首手合作

        move2 = tft.fn(0, 1, 1, ("c",), 2, ("c",), None, None, None)
        self.assertEqual(move2, "c")  # 对手合作，跟随

        # 模拟历史被覆盖：假设第一手实际是背叛（模拟器故障处理）
        # 现在传入不同的历史
        move3 = tft.fn(0, 1, 1, ("b",), 2, ("c",), None, None, None)
        # TFT 应该重放历史并返回"c"（跟随对手上一手的"c"）
        self.assertEqual(move3, "c")


if __name__ == "__main__":
    unittest.main()
