"""训练环境：并行 mini-锦标赛与轨迹采集（issue #6，sub-issue 8.2）。

课程每轮所有小组两两对局；本环境用一个"训练赛季"近似：

- 每轮从对手池无放回抽取 `opponents_per_round` 个对手，策略与它们**并行**对局
  （同一回合内所有对局同时决策）；
- 每个对手维护自己的最近 20 场历史，策略也维护自己的最近 20 场历史；
  历史视图在每轮开始时按 `TournamentConfig` 生成，整轮不变，轮末统一追加
  （等价于模拟器的 `history_update="round"`，见 docs/roadmap.md 待确认问题 1）；
- 对手函数是纯函数，所以同一轮的对局互不影响，可以整批采样动作；
- 轨迹按"场"保存：历史视图只存一份索引张量，本场动作按前缀重建，避免
  每个回合都复制 20 × 25 的历史。

奖励默认就是单回合收益之和（即本场得分）；可选信誉塑形项（ADR 0004）：
`shaping * 无端背叛次数`，无端背叛定义为"我方背叛且对手上一手合作"。

本模块不更新网络参数，也不改变 policy 的 train/eval 模式；这两件事由
`ipd/training/rl_train.py` 负责。
"""

from collections import deque
from dataclasses import dataclass, field
import random

import torch
from torch.distributions import Categorical

from ipd.agents.encoder import moves_to_indices
from ipd.sim import rules
from ipd.sim.config import TournamentConfig
from ipd.sim.tournament import history_view
from ipd.training.policy import action_to_move

DEFAULT_OPPONENTS_PER_ROUND = 30


@dataclass
class MatchTrajectory:
    """一场 25 回合对局中策略侧的轨迹。"""

    opponent: str
    opponent_id: int
    opponent_index: int
    rs: int
    # 策略看到的对手历史视图，已转成编码器索引 (20, 25)。
    own_history: torch.Tensor
    opp_history: torch.Tensor
    actions: list = field(default_factory=list)        # 策略动作索引 0/1
    opp_moves: list = field(default_factory=list)      # 对手动作 "c"/"b"
    payoffs: list = field(default_factory=list)        # 原始单回合得分
    rewards: list = field(default_factory=list)        # 训练奖励（可含塑形）
    log_probs: list = field(default_factory=list)
    entropies: list = field(default_factory=list)
    values: list = field(default_factory=list)
    # 教师标签（DAgger 用）；为空表示"行为动作即标签"（教师自己跑的轨迹）。
    labels: list = field(default_factory=list)

    @property
    def score(self):
        """本场原始得分（不含塑形项）。"""
        return sum(self.payoffs)

    @property
    def action_moves(self):
        """本场策略动作字符串序列。"""
        return [action_to_move(action) for action in self.actions]


@dataclass
class Episode:
    """一个训练赛季的轨迹。"""

    matches: list
    rounds: int
    seed: int
    policy_id: int = 1

    @property
    def transitions(self):
        return sum(len(match.actions) for match in self.matches)

    @property
    def mean_score(self):
        if not self.matches:
            return 0.0
        return sum(match.score for match in self.matches) / len(self.matches)

    def scores_by_opponent(self):
        """按对手名汇总每场平均分。"""
        totals = {}
        counts = {}
        for match in self.matches:
            totals[match.opponent] = totals.get(match.opponent, 0.0) + match.score
            counts[match.opponent] = counts.get(match.opponent, 0) + 1
        return {name: totals[name] / counts[name] for name in totals}


class TrainingEnv:
    """并行采集策略轨迹的 mini-锦标赛环境。

    Args:
        agents: 对手池（`ipd.sim.Agent`），名字唯一。
        cfg: 锦标赛配置；默认关闭超时（训练不受 0.5 s 限制）。
        seed: 环境随机种子（对手抽样、历史隐藏、对手的全局 random）。
        opponents_per_round: 每轮抽取的对手数；None 表示 min(池大小, 30)。

    Raises:
        ValueError: 对手池为空、名字重复或 opponents_per_round 非法。
    """

    def __init__(self, agents, cfg=None, seed=0, opponents_per_round=None):
        agents = list(agents)
        if not agents:
            raise ValueError("training opponent pool must not be empty")
        names = [agent.name for agent in agents]
        if len(set(names)) != len(names):
            raise ValueError("opponent names must be unique")
        self.agents = agents
        self.cfg = cfg or TournamentConfig(timeout=None, shuffle_schedule=False)
        self.seed = int(seed)
        limit = len(agents) if opponents_per_round is None else int(opponents_per_round)
        if limit <= 0:
            raise ValueError("opponents_per_round must be positive")
        self.opponents_per_round = min(limit, len(agents))
        self.policy_id = 1
        self._rng = random.Random(f"train-env-{self.seed}")
        self._opponent_ids = list(range(2, len(agents) + 2))
        self._policy_history = deque(maxlen=self.cfg.history_matches)
        self._opponent_histories = [
            deque(maxlen=self.cfg.history_matches) for _ in agents
        ]
        self._view_rng = random.Random(f"train-view-{self.seed}")
        self.reset()

    def reset(self, seed=None):
        """清空历史并重置随机流；seed=None 时沿用构造时的种子。"""
        if seed is not None:
            self.seed = int(seed)
            self._rng = random.Random(f"train-env-{self.seed}")
            self._view_rng = random.Random(f"train-view-{self.seed}")
        self._policy_history.clear()
        for history in self._opponent_histories:
            history.clear()
        # 对手策略使用全局 random；在每次赛季开始时固定，保证可复现。
        random.seed(self.seed)

    def _start_round(self, rs, indices):
        """构造一轮的并行对局：历史视图在轮首生成，整轮不变。"""
        matches = []
        for index in indices:
            opponent = self.agents[index]
            # 策略看到的是对手的历史（对手自己的动作 + 对手的对手的动作）。
            view_own, view_opp = history_view(
                self._opponent_histories[index], self.cfg, self._view_rng
            )
            # 对手看到的是策略的历史；模拟器同样为每场单独生成隐藏掩码。
            opponent_own, opponent_opp = history_view(
                self._policy_history, self.cfg, self._view_rng
            )
            match = MatchTrajectory(
                opponent=opponent.name,
                opponent_id=self._opponent_ids[index],
                opponent_index=index,
                rs=rs,
                own_history=moves_to_indices(view_own),
                opp_history=moves_to_indices(view_opp),
            )
            matches.append((match, opponent, opponent_own, opponent_opp))
        return matches

    def rollout(self, policy, shaping=0.0, generator=None, labeler=None):
        """跑一个完整赛季（cfg.rounds 轮），返回 `Episode`。

        Args:
            policy: `IPDPolicy`（只前向采样，不更新参数）。
            shaping: 无端背叛惩罚系数 λ；0 表示纯单场得分。
            generator: torch 随机数生成器，控制动作采样。
            labeler: 可选的教师策略（实现 `act_with_value`）；给出时为每个状态
                记录教师动作到 `match.labels`，供 RL 的模仿正则项使用。

        Returns:
            `Episode`。

        Raises:
            ValueError: shaping 为负。
        """
        if shaping < 0:
            raise ValueError("shaping must be non-negative")
        matches = []
        for rs in range(self.cfg.rounds):
            indices = self._rng.sample(
                range(len(self.agents)), self.opponents_per_round
            )
            round_items = self._start_round(rs, indices)
            own_batch = torch.stack([item[0].own_history for item in round_items])
            opp_batch = torch.stack([item[0].opp_history for item in round_items])
            rs_batch = torch.full(
                (len(round_items),), rs, dtype=torch.long, device=own_batch.device
            )
            for length in range(rules.TURNS):
                ours = [item[0].actions for item in round_items]
                theirs = [item[0].opp_moves for item in round_items]
                my_tensor = self._decisions_tensor(ours, offset=1)
                opp_tensor = self._move_tensor(theirs)
                with torch.no_grad():
                    actions, probs, values = policy.act_with_value(
                        own_batch,
                        opp_batch,
                        my_tensor,
                        opp_tensor,
                        rs_batch,
                        length,
                        generator=generator,
                    )
                distribution = Categorical(probs=probs)
                log_probs = distribution.log_prob(actions)
                entropies = distribution.entropy()
                labels = None
                if labeler is not None:
                    label_actions, _, _ = labeler.act_with_value(
                        own_batch,
                        opp_batch,
                        my_tensor,
                        opp_tensor,
                        rs_batch,
                        length,
                    )
                    labels = label_actions.tolist()
                for position, (match, opponent, opp_own, opp_opp) in enumerate(round_items):
                    our_move = action_to_move(int(actions[position].item()))
                    # 对手只能看到 length 之前的动作，与课程接口一致。
                    their_moves_before = tuple(match.opp_moves)
                    our_moves_before = tuple(match.action_moves)
                    opponent_move = self._opponent_move(
                        opponent,
                        match,
                        rs,
                        length,
                        our_moves_before,
                        their_moves_before,
                        opp_own,
                        opp_opp,
                    )
                    payoff = rules.payoff(our_move, opponent_move, rules.REPAY[length])
                    reward = float(payoff)
                    if (
                        shaping
                        and length > 0
                        and our_move == rules.DEFECT
                        and match.opp_moves[-1] == rules.COOPERATE
                    ):
                        reward -= shaping
                    match.actions.append(int(actions[position].item()))
                    match.opp_moves.append(opponent_move)
                    match.payoffs.append(payoff)
                    match.rewards.append(reward)
                    match.log_probs.append(float(log_probs[position].item()))
                    match.entropies.append(float(entropies[position].item()))
                    if values is not None:
                        match.values.append(float(values[position].item()))
                    else:
                        match.values.append(0.0)
                    if labels is not None:
                        match.labels.append(int(labels[position]))
            # 轮末统一写入历史（history_update="round"）。
            for match, opponent, _, _ in round_items:
                their_moves = tuple(match.opp_moves)
                our_moves = tuple(match.action_moves)
                self._policy_history.append((our_moves, their_moves))
                self._opponent_histories[match.opponent_index].append(
                    (their_moves, our_moves)
                )
            matches.extend(item[0] for item in round_items)
        return Episode(matches=matches, rounds=self.cfg.rounds, seed=self.seed)

    @staticmethod
    def _decisions_tensor(actions_per_match, offset=1):
        """把每场动作索引列表转成编码器输入 (batch, length)。"""
        rows = [[action + offset for action in actions] for actions in actions_per_match]
        if not rows:
            return torch.zeros((0, 0), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)

    @staticmethod
    def _move_tensor(moves_per_match):
        """把每场对手动作转成编码器索引 (batch, length)：c=1, b=2。"""
        rows = [
            [1 if move == rules.COOPERATE else 2 for move in moves]
            for moves in moves_per_match
        ]
        if not rows:
            return torch.zeros((0, 0), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)

    def _opponent_move(
        self, opponent, match, rs, length, our_moves, their_moves, opp_own, opp_opp
    ):
        """调用对手策略；非法返回值/异常按合作处理（与模拟器一致）。"""
        try:
            move = opponent.fn(
                rs,
                length,
                match.opponent_id,
                their_moves,
                self.policy_id,
                our_moves,
                rules.REPAY,
                opp_own,
                opp_opp,
            )
        except Exception:
            return rules.COOPERATE
        return move if move in rules.LEGAL_MOVES else rules.COOPERATE


__all__ = ["Episode", "MatchTrajectory", "TrainingEnv"]
