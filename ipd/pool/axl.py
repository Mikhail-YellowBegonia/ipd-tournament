"""Axelrod 策略库适配器（仅训练端，依赖 axelrod 4.14.0）。

将 axelrod 中的策略包装为我方九参数接口的 Agent，layer 为 "AXL"。

要点：
- axelrod 的 Player 有内部状态，并假设每回合都调用过 strategy()。
  因此每场维护一个实例；需要重建时逐回合"调用 strategy() 再写入实际动作"重放，
  与 axl.Match.simultaneous_play 的顺序一致，写入的是模拟器记录的实际动作
  （可能与策略本意不同，例如故障被改为合作）。
- axelrod 的 Game 在整场内固定，我方收益分阶段变化。三个阶段的大小顺序
  T > R > P > S 相同，且满足 2R > T + S，选用第 0 阶段 (T, S, R, P) = (4, -6, 2, -1)。
- 少数策略（零行列式 ZD 系，及以它们为成员的 Meta*MemoryOne）由收益推出概率向量，
  在第 0 阶段收益下概率越界、无法构造，这些策略退回 axelrod 默认收益 (5, 0, 3, 1)。
- 排除作弊策略（inspects_source / manipulates_source / manipulates_state）；
  long_run_time 策略需显式包含。
- 不吞异常：策略出错交给模拟器按故障处理，测试据此发现问题。
"""

import random
from typing import Optional

import axelrod as axl

from ipd.sim import rules
from ipd.sim.agent import Agent

GAME = axl.Game(r=2, s=-6, t=4, p=-1)

_TO_AXL = {rules.COOPERATE: axl.Action.C, rules.DEFECT: axl.Action.D}
_FROM_AXL = {axl.Action.C: rules.COOPERATE, axl.Action.D: rules.DEFECT}


def _game_for(cls):
    """优先用第 0 阶段收益；策略无法接受时退回默认收益。"""
    try:
        cls().set_match_attributes(length=rules.TURNS, game=GAME)
        return GAME
    except ValueError:
        return axl.DefaultGame


def _new_pair(cls, game, seed):
    player = cls()
    # 镜像对手：只用来承载我方对手的历史
    mirror = axl.Cooperator()
    for p in (player, mirror):
        p.set_match_attributes(length=rules.TURNS, game=game)
    player.set_seed(seed)
    return player, mirror


def _make_fn(cls, game):
    # 每场一个实例：(rs, my_id, opp_id) -> 状态字典
    matches = {}

    def replay(st, my, opp):
        player, mirror = _new_pair(cls, game, st["seed"])
        for t in range(len(my)):
            player.strategy(mirror)
            player.update_history(_TO_AXL[my[t]], _TO_AXL[opp[t]])
            mirror.update_history(_TO_AXL[opp[t]], _TO_AXL[my[t]])
        st.update(player=player, mirror=mirror)

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        key = (rs, my_id, opp_id)
        st = matches.get(key)
        if length == 0 or st is None:
            seed = random.randrange(2**31)
            player, mirror = _new_pair(cls, game, seed)
            st = matches[key] = {"player": player, "mirror": mirror, "seed": seed,
                                 "my": (), "opp": (), "called": 0}
        if length:
            if st["called"] == length and my[:-1] == st["my"] and opp[:-1] == st["opp"]:
                # 正常路径：上一回合已调用 strategy()，只需写入实际动作
                st["player"].update_history(_TO_AXL[my[-1]], _TO_AXL[opp[-1]])
                st["mirror"].update_history(_TO_AXL[opp[-1]], _TO_AXL[my[-1]])
            else:
                # 记录与实例不一致，用相同种子从头重放
                replay(st, my, opp)
            st["my"], st["opp"] = my, opp

        move = _FROM_AXL[st["player"].strategy(st["mirror"])]
        st["called"] = length + 1
        if length == rules.TURNS - 1:
            del matches[key]
        return move
    return s


def _excluded(cls, include_long_run_time):
    c = cls().classifier
    if c.get("inspects_source") or c.get("manipulates_source") or c.get("manipulates_state"):
        return "cheating"
    if c.get("long_run_time") and not include_long_run_time:
        return "long_run_time"
    return None


def axelrod_agents(include_long_run_time=False, names: Optional[list] = None):
    """返回适配后的 axelrod 策略。names 为类名列表，None 表示全部。"""
    agents = []
    for cls in axl.all_strategies:
        if names is not None and cls.__name__ not in names:
            continue
        if _excluded(cls, include_long_run_time):
            continue
        game = _game_for(cls)
        params = {} if game is GAME else {"game": "default"}
        agents.append(Agent(name=f"axl:{cls.__name__}", fn=_make_fn(cls, game),
                            family=cls.__name__, layer="AXL", params=params))
    return agents


def excluded_strategies(include_long_run_time=False):
    """返回 {类名: 排除原因}。"""
    out = {}
    for cls in axl.all_strategies:
        reason = _excluded(cls, include_long_run_time)
        if reason:
            out[cls.__name__] = reason
    return out
