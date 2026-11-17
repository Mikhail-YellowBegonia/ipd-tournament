"""策略网络（issue #6，sub-issue 8.1）：历史编码器 + 动作头（+ 价值头）。

动作索引沿用 issue 的约定：`0 = 合作 (c)`、`1 = 背叛 (b)`；编码器内部使用
`m=0, c=1, b=2`，两者只在 `observation_tensors()` 与 `action_to_move()` 处转换。

网络结构::

    HistoryEncoder -> state_embed -> Linear(embed_dim, 2)     动作 logits
                                  -> Linear(embed_dim, 1)     价值（可选，REINFORCE 基线）

本模块只定义网络与单步推理；轨迹采集见 `ipd/training/env.py`，训练循环见
`ipd/training/rl_train.py`。
"""

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from ipd.agents.encoder import HistoryEncoder, observation_tensors
from ipd.sim.agent import Agent
from ipd.sim import rules

COOPERATE_INDEX = 0
DEFECT_INDEX = 1
ACTION_TO_MOVE = ("c", "b")
MOVE_TO_ACTION = {"c": COOPERATE_INDEX, "b": DEFECT_INDEX}


def action_to_move(action):
    """把策略的动作索引（0/1）转成课程接口的 `"c"`/`"b"`。"""
    return ACTION_TO_MOVE[int(action)]


def move_to_action(move):
    """把 `"c"`/`"b"` 转成动作索引（0/1）。"""
    return MOVE_TO_ACTION[move]


class IPDPolicy(nn.Module):
    """编码器 + 动作头；可选价值头用于 REINFORCE 基线。

    Args:
        encoder: 已实例化的编码器；None 时用 `HistoryEncoder(**encoder_kwargs)`。
        action_dim: 动作数，固定为 2（合作/背叛）。
        use_value: 是否创建价值头（训练时用，纯采样可关）。
        encoder_kwargs: 传给 `HistoryEncoder` 的参数（仅 encoder=None 时使用）。
    """

    def __init__(self, encoder=None, action_dim=2, use_value=True, encoder_kwargs=None):
        super().__init__()
        if action_dim != 2:
            raise ValueError("IPDPolicy only supports action_dim=2 (cooperate/defect)")
        if encoder is None:
            encoder = HistoryEncoder(**(encoder_kwargs or {}))
        if not hasattr(encoder, "embed_dim") or not hasattr(encoder, "forward"):
            raise TypeError("encoder must expose embed_dim and forward()")
        self.encoder = encoder
        self.action_dim = action_dim
        self.embed_dim = encoder.embed_dim
        self.action_head = nn.Linear(self.embed_dim, action_dim)
        self.value_head = nn.Linear(self.embed_dim, 1) if use_value else None

    @property
    def use_value(self):
        return self.value_head is not None

    def num_parameters(self):
        """可训练参数数量（含编码器、动作头、价值头）。"""
        return sum(parameter.numel() for parameter in self.parameters())

    def config(self):
        """构造参数快照，供检查点与重建使用。"""
        encoder_config = {
            "embed_dim": getattr(self.encoder, "embed_dim", None),
            "max_history": getattr(self.encoder, "max_history", None),
            "turns": getattr(self.encoder, "turns", None),
            "num_heads": getattr(self.encoder, "num_heads", None),
            "dropout": getattr(self.encoder, "dropout", None),
            "decay_half_life": getattr(self.encoder, "decay_half_life", None),
            "max_rounds": getattr(self.encoder, "max_rounds", None),
        }
        return {
            "action_dim": self.action_dim,
            "use_value": self.use_value,
            "encoder": encoder_config,
        }

    @classmethod
    def from_config(cls, config):
        """按 `config()` 的快照重建网络结构（不加载权重）。"""
        encoder_config = dict(config.get("encoder", {}))
        known = {
            key: encoder_config[key]
            for key in (
                "embed_dim",
                "max_history",
                "turns",
                "num_heads",
                "dropout",
                "decay_half_life",
                "max_rounds",
            )
            if encoder_config.get(key) is not None
        }
        return cls(
            encoder=None,
            action_dim=config.get("action_dim", 2),
            use_value=config.get("use_value", True),
            encoder_kwargs=known,
        )

    def encode(self, own_history, opp_history, my_decisions, opp_decisions, rs, length):
        """返回状态嵌入 `(batch, embed_dim)`。"""
        return self.encoder(own_history, opp_history, my_decisions, opp_decisions, rs, length)

    def forward(self, own_history, opp_history, my_decisions, opp_decisions, rs, length):
        """返回动作 logits `(batch, 2)`：第 0 列合作、第 1 列背叛。"""
        state = self.encode(own_history, opp_history, my_decisions, opp_decisions, rs, length)
        return self.action_head(state)

    def forward_with_value(
        self, own_history, opp_history, my_decisions, opp_decisions, rs, length
    ):
        """返回 `(logits, values)`；`values` 形状 `(batch,)`。"""
        state = self.encode(own_history, opp_history, my_decisions, opp_decisions, rs, length)
        logits = self.action_head(state)
        if self.value_head is None:
            return logits, None
        return logits, self.value_head(state).squeeze(-1)

    @staticmethod
    def _sample(probs, deterministic=False, generator=None):
        if deterministic:
            return probs.argmax(dim=-1)
        if generator is None:
            return Categorical(probs=probs).sample()
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

    def act(self, *encoder_inputs, deterministic=False, generator=None):
        """采样动作（issue 接口）。

        Args:
            *encoder_inputs: 六个编码器输入（历史、本场动作、rs、length）。
            deterministic: True 时取 argmax，否则按 softmax 概率采样。
            generator: 可选的 torch 随机数生成器，控制采样可复现。

        Returns:
            `(action, probs)`；`action` 形状 `(batch,)`，`probs` 形状 `(batch, 2)`。
        """
        logits = self.forward(*encoder_inputs)
        probs = F.softmax(logits, dim=-1)
        action = self._sample(probs, deterministic=deterministic, generator=generator)
        return action, probs

    def act_with_value(self, *encoder_inputs, deterministic=False, generator=None):
        """采样动作并返回价值，供训练环境一次前向同时拿到三者。

        Returns:
            `(action, probs, values)`；`values` 为 None 表示没有价值头。
        """
        logits, values = self.forward_with_value(*encoder_inputs)
        probs = F.softmax(logits, dim=-1)
        action = self._sample(probs, deterministic=deterministic, generator=generator)
        return action, probs, values

    def distribution(self, *encoder_inputs):
        """返回动作分布，供策略梯度使用。"""
        logits = self.forward(*encoder_inputs)
        return Categorical(logits=logits)


def policy_agent(policy, name="policy"):
    """把策略网络包装成模拟器可用的确定性 `Agent`（评估/验收用）。

    每次调用独立编码，不保存任何跨调用状态；异常时按规则返回合作。
    """
    policy.eval()

    def strategy(
        rs,
        length,
        my_id,
        my_decisions,
        opp_id,
        opp_decisions,
        repay,
        opp_history_decisions,
        opp_opp_history_decisions,
    ):
        try:
            inputs = observation_tensors(
                opp_history_decisions,
                opp_opp_history_decisions,
                my_decisions,
                opp_decisions,
                rs,
            )
            with torch.no_grad():
                logits = policy(*inputs, int(length))
                action = int(logits.argmax(dim=-1).item())
            return action_to_move(action)
        except Exception:
            return rules.COOPERATE

    return Agent(name=name, fn=strategy, family="policy", layer="RL")


__all__ = [
    "ACTION_TO_MOVE",
    "COOPERATE_INDEX",
    "DEFECT_INDEX",
    "MOVE_TO_ACTION",
    "IPDPolicy",
    "action_to_move",
    "move_to_action",
    "policy_agent",
]
