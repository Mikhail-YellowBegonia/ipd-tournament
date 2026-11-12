"""Axelrod 策略库适配器。

将 Axelrod 策略库（axelrod）中的策略适配为我方模拟器接口。

要点：
- Axelrod 的 Player 对象有状态，需要为每场对局维护独立实例
- Axelrod 使用 'C'/'D'，我方使用 'c'/'b'
- Axelrod 的 Game 收益矩阵固定，我方分阶段变化；选用第 0 阶段收益作为映射
- 排除作弊策略（inspects_source / manipulates_source / manipulates_state）
- long_run_time 策略需显式包含
- 捕获构造或运行异常，排除有问题的策略
"""

import random
from typing import Optional

try:
    import axelrod as axl
    AXELROD_AVAILABLE = True
except ImportError:
    AXELROD_AVAILABLE = False

from ipd.sim.agent import Agent

# 我方使用第 0 阶段收益 (T=4, S=-6, R=2, P=-1) 作为 Axelrod Game 映射
# 理由：第 0 阶段是开局，占比最大（10/25 回合），且策略初始化时就需要 Game
_GAME = None if not AXELROD_AVAILABLE else axl.Game(r=2, s=-6, t=4, p=-1)

_MATCH_LENGTH = 25


def _is_cheating(strategy_class):
    """判断策略是否作弊（检查或操纵源码/状态）。"""
    classifier = strategy_class().classifier
    return any([
        classifier.get("inspects_source", False),
        classifier.get("manipulates_source", False),
        classifier.get("manipulates_state", False),
    ])


def _is_long_run_time(strategy_class):
    """判断策略是否被标记为长运行时间。"""
    return strategy_class().classifier.get("long_run_time", False)


def _can_construct(strategy_class):
    """尝试构造策略实例，返回是否成功。"""
    try:
        strategy_class()
        return True
    except Exception:
        return False


def _make_axelrod_fn(strategy_class):
    """为 Axelrod 策略类生成我方接口的策略函数。

    状态管理：
    - 每场对局（由 rs, my_id, opp_id 唯一标识）维护独立的 player 实例
    - length==0 时创建新实例
    - 每次调用时根据实际历史同步 player 状态（处理模拟器可能覆盖动作的情况）

    关键：Axelrod 的 strategy() 方法需要对手对象来读取对手历史，
    但不会自动更新历史。我们在每次调用时根据实际历史同步 player 状态。
    """
    # 闭包中保存每场对局的状态
    cache = {}

    def strategy_fn(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        # 用 (rs, my_id, opp_id) 标识唯一对局
        match_key = (rs, my_id, opp_id)

        # 如果是对局开始，创建新实例
        if length == 0:
            player = strategy_class()
            opponent = axl.Cooperator()  # 镜像对手，只记录历史

            # 设置匹配属性
            player.match_attributes = {
                "length": _MATCH_LENGTH,
                "game": _GAME,
            }
            opponent.match_attributes = {
                "length": _MATCH_LENGTH,
                "game": _GAME,
            }

            # 从全局 random 模块为 Axelrod 策略设置种子
            # 这确保了确定性策略的可重现性
            if hasattr(player, "_random"):
                seed = random.randint(0, 2**31 - 1)
                player._random = axl.RandomGenerator(seed)

            cache[match_key] = {
                "player": player,
                "opponent": opponent,
            }

        state = cache.get(match_key)
        if state is None:
            # 异常情况：缓存丢失，应该不会发生但做防御
            # 创建新实例并在下面重建历史
            player = strategy_class()
            opponent = axl.Cooperator()
            player.match_attributes = {"length": _MATCH_LENGTH, "game": _GAME}
            opponent.match_attributes = {"length": _MATCH_LENGTH, "game": _GAME}
            if hasattr(player, "_random"):
                seed = random.randint(0, 2**31 - 1)
                player._random = axl.RandomGenerator(seed)
            state = cache[match_key] = {
                "player": player,
                "opponent": opponent,
            }
        else:
            player = state["player"]
            opponent = state["opponent"]

        # 同步历史：确保 player 的历史与实际历史一致
        # 检查当前历史长度是否匹配
        if len(player.history) != length:
            # 历史长度不匹配，需要重建
            player.reset()
            opponent.reset()
            player.match_attributes = {"length": _MATCH_LENGTH, "game": _GAME}
            opponent.match_attributes = {"length": _MATCH_LENGTH, "game": _GAME}

            # 重放实际历史
            for i in range(length):
                # 转换我方记录：'c' -> axl.Action.C, 'b' -> axl.Action.D
                my_action = axl.Action.C if my[i] == "c" else axl.Action.D
                opp_action = axl.Action.C if opp[i] == "c" else axl.Action.D

                # 更新双方历史：append(my_action, opponent_action)
                player.history.append(my_action, opp_action)
                opponent.history.append(opp_action, my_action)

        # 调用 Axelrod 策略获取动作
        try:
            action = player.strategy(opponent)
        except Exception:
            # 策略执行异常，返回合作
            return "c"

        # 转换为我方格式
        result = "c" if action == axl.Action.C else "b"

        return result

    return strategy_fn


def axelrod_agents(
    include_long_run_time: bool = False,
    names: Optional[list] = None,
) -> list[Agent]:
    """返回可用的 Axelrod 策略列表，适配为我方 Agent 接口。

    参数：
        include_long_run_time: 是否包含长运行时间策略（默认 False）
        names: 如果指定，只包含这些名称的策略；否则包含所有可用策略

    返回：
        Agent 列表，layer="AXL"，family=策略类名

    排除：
        - 作弊策略（inspects_source / manipulates_source / manipulates_state）
        - 无法构造的策略
        - long_run_time 策略（除非 include_long_run_time=True）
        - 运行时崩溃的策略
    """
    if not AXELROD_AVAILABLE:
        return []

    agents = []
    all_strategies = axl.all_strategies

    # 如果指定了名称，过滤策略
    if names is not None:
        name_set = set(names)
        all_strategies = [s for s in all_strategies if s.__name__ in name_set]

    for strategy_class in all_strategies:
        # 排除作弊策略
        if _is_cheating(strategy_class):
            continue

        # 排除长运行时间策略（除非明确包含）
        if not include_long_run_time and _is_long_run_time(strategy_class):
            continue

        # 排除无法构造的策略
        if not _can_construct(strategy_class):
            continue

        # 创建适配的策略函数
        fn = _make_axelrod_fn(strategy_class)

        # 创建 Agent
        name = strategy_class.__name__
        agent = Agent(
            name=name,
            fn=fn,
            family=name,
            layer="AXL",
            params={},
        )
        agents.append(agent)

    return agents
