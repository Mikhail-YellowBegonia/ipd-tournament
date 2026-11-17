"""PyTorch 历史编码器（issue #5，ADR 0002）。

输入与语义
----------
课程接口只暴露**对手的 20 场历史**，不提供我方自己的历史。因此 `forward` 的
前两个参数沿用 issue 中的名字，但语义是历史主体的两份张量：

- `own_history`：历史主体（对手）自己的动作，对应 `opp_history_decisions`；
- `opp_history`：该主体当时的对手的动作，对应 `opp_opp_history_decisions`。

索引统一为 `m=0, c=1, b=2`；当前场 `my_decisions/opp_decisions` 也用同一套索引
（不会有 `m`）。形状要求：

- 历史：`(batch, 20, 25)` 的 `torch.long`；
- 本场：`(batch, length)` 的 `torch.long`；
- `rs`：`0–7` 的整数或 `(batch,)` 张量；
- `length`：标量整数（本场已完成回合数）。

编码流程
--------
1. 场内：每场按（主体动作, 对方动作, 收益阶段）做 27 词表 n-gram 计数，
   除以有效步数后乘可学习嵌入表，得到 20 个场向量；含 `m` 的步跳过。
2. 场间：20 个场向量做带时间衰减和有效场掩码的多头注意力池化；全 `m`
   的场不参与注意力，全空历史返回零向量。
3. 本场：GRU 读本场动作对（复用同一张 n-gram 嵌入表），初始隐状态由历史
   嵌入与当前轮次嵌入给出；输出与本场嵌入残差相加并 LayerNorm。

性能目标：CPU、batch=1、完整历史下单次前向 ≤ 1 ms。`tests/test_encoder.py`
中的基准测试会报告最坏/中位耗时。
"""

import torch
from torch import nn
import torch.nn.functional as F

from ipd.sim import rules

# 动作 -> 索引；历史与本场共用，m 只出现在历史中。
MOVE_TO_INDEX = {"m": 0, "c": 1, "b": 2}
INDEX_TO_MOVE = {index: move for move, index in MOVE_TO_INDEX.items()}
VOCAB_SIZE = 27  # 3 (主体) × 3 (对方) × 3 (阶段)
NEXT_MOVE_TARGET = {"c": 0, "b": 1}
NEG_INF = -1e9


def moves_to_indices(rows, device=None):
    """把 `c/b/m` 字符串行转成 `long` 张量；保持行数。"""
    data = [[MOVE_TO_INDEX[move] for move in row] for row in rows]
    return torch.tensor(data, dtype=torch.long, device=device)


def observation_tensors(
    opp_history_decisions,
    opp_opp_history_decisions,
    my_decisions,
    opp_decisions,
    rs,
    device=None,
):
    """把一次考试接口观测转成编码器输入（都带 batch 维）。

    Returns:
        `(own_history, opp_history, my_decisions, opp_decisions, rs)`；
        历史形状 `(1, 20, 25)`，本场形状 `(1, length)`，`rs` 形状 `(1,)`。
    """
    own = moves_to_indices(opp_history_decisions, device=device).unsqueeze(0)
    opp = moves_to_indices(opp_opp_history_decisions, device=device).unsqueeze(0)
    my = moves_to_indices((tuple(my_decisions),), device=device)
    their = moves_to_indices((tuple(opp_decisions),), device=device)
    if isinstance(rs, torch.Tensor):
        round_id = rs.to(dtype=torch.long, device=device).reshape(1)
    else:
        round_id = torch.tensor([rs], dtype=torch.long, device=device)
    return own, opp, my, their, round_id


class HistoryEncoder(nn.Module):
    """分层历史编码器，输出固定维度的状态嵌入。"""

    def __init__(
        self,
        embed_dim=32,
        max_history=rules.HISTORY_MATCHES,
        turns=rules.TURNS,
        num_heads=4,
        dropout=0.0,
        decay_half_life=10.0,
        max_rounds=rules.ROUNDS,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if max_history <= 0 or turns <= 0:
            raise ValueError("max_history and turns must be positive")
        if decay_half_life <= 0:
            raise ValueError("decay_half_life must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embed_dim = embed_dim
        self.max_history = max_history
        self.turns = turns
        self.num_heads = num_heads
        self.max_rounds = max_rounds
        self.vocab_size = VOCAB_SIZE

        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        # 可学习的池化查询向量；多头注意力内部会再分头。
        self.match_query = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.normal_(self.match_query, std=0.02)
        self.match_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.history_proj = nn.Linear(embed_dim, embed_dim)
        self.round_embed = nn.Embedding(max_rounds, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(embed_dim)

        phase = torch.tensor(
            [rules.phase_of(t) for t in range(turns)], dtype=torch.long
        )
        decay = torch.tensor(
            [
                0.5 ** ((max_history - 1 - i) / decay_half_life)
                for i in range(max_history)
            ],
            dtype=torch.float32,
        )
        self.register_buffer("phase_of_turn", phase, persistent=False)
        self.register_buffer("time_decay", decay, persistent=False)
        self.register_buffer("log_time_decay", torch.log(decay), persistent=False)

    def num_parameters(self):
        """可训练参数数量（用于 #5/#6 的容量记录）。"""
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_history(self, own_history, opp_history):
        for name, tensor in (("own_history", own_history), ("opp_history", opp_history)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.dtype != torch.long:
                raise TypeError(f"{name} must be torch.long, got {tensor.dtype}")
            if tensor.ndim != 3:
                raise ValueError(f"{name} must have shape (batch, 20, 25)")
            if tensor.shape[1:] != (self.max_history, self.turns):
                raise ValueError(
                    f"{name} must have shape (batch, {self.max_history}, {self.turns}), "
                    f"got {tuple(tensor.shape)}"
                )
        if own_history.shape[0] != opp_history.shape[0]:
            raise ValueError("own_history and opp_history must have the same batch size")
        return own_history, opp_history

    def _validate_current(self, my_decisions, opp_decisions, batch_size, length):
        for name, tensor in (
            ("my_decisions", my_decisions),
            ("opp_decisions", opp_decisions),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.dtype != torch.long:
                raise TypeError(f"{name} must be torch.long, got {tensor.dtype}")
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape (batch, length)")
            if tensor.shape[0] != batch_size:
                raise ValueError(f"{name} batch size must be {batch_size}")
            if tensor.shape[1] > self.turns:
                raise ValueError(f"{name} has more than {self.turns} turns")
        if my_decisions.shape != opp_decisions.shape:
            raise ValueError("my_decisions and opp_decisions must have the same shape")

        if isinstance(length, torch.Tensor):
            if length.numel() != 1:
                raise ValueError("length must be a scalar")
            length = int(length.item())
        if not isinstance(length, int):
            raise TypeError("length must be an int or a scalar tensor")
        if not 0 <= length <= self.turns:
            raise ValueError(f"length must be in [0, {self.turns}], got {length}")
        if length > my_decisions.shape[1]:
            raise ValueError(
                f"length={length} exceeds the provided decision length "
                f"{my_decisions.shape[1]}"
            )
        return length

    def _round_ids(self, rs, batch_size, device):
        if isinstance(rs, torch.Tensor):
            if rs.ndim == 0:
                rs = rs.repeat(batch_size)
            if rs.ndim != 1 or rs.shape[0] != batch_size:
                raise ValueError("rs must be a scalar or a (batch,) tensor")
            rs = rs.to(device=device, dtype=torch.long)
        else:
            if not isinstance(rs, int):
                raise TypeError("rs must be an int or a tensor")
            rs = torch.full((batch_size,), rs, dtype=torch.long, device=device)
        if bool((rs < 0).any()) or bool((rs >= self.max_rounds).any()):
            raise ValueError(f"rs must be in [0, {self.max_rounds - 1}]")
        return rs

    def encode_matches(self, own_history, opp_history):
        """把 20 场历史编码成场向量。

        Returns:
            `(match_embeds, valid)`，形状 `(batch, 20, embed_dim)` 与
            `(batch, 20)`；全 `m` 的场 `valid=False`。
        """
        self._validate_history(own_history, opp_history)
        valid_step = (own_history != 0) & (opp_history != 0)
        phase = self.phase_of_turn[None, None, :]
        indices = own_history * 9 + opp_history * 3 + phase
        one_hot = F.one_hot(indices, self.vocab_size).to(self.embed.weight.dtype)
        one_hot = one_hot * valid_step.unsqueeze(-1).to(one_hot.dtype)
        counts = one_hot.sum(dim=2)
        steps = valid_step.sum(dim=2, keepdim=True).clamp_min(1).to(counts.dtype)
        counts = counts / steps
        match_embeds = counts @ self.embed.weight
        valid = valid_step.any(dim=2)
        return match_embeds, valid

    def pool_matches(self, match_embeds, valid, return_weights=False):
        """带时间衰减与掩码的注意力池化。

        Args:
            match_embeds: `(batch, 20, embed_dim)`。
            valid: `(batch, 20)` 的布尔张量；全 `m` 的场为 False。
            return_weights: 是否返回平均后的注意力权重。

        Returns:
            `(pooled, weights)`；`pooled` 为 `(batch, embed_dim)`，全空历史的行为零；
            `weights` 为 `(batch, 1, 20)` 或 `None`。
        """
        if not isinstance(match_embeds, torch.Tensor):
            raise TypeError("match_embeds must be a torch.Tensor")
        if match_embeds.ndim != 3:
            raise ValueError("match_embeds must have shape (batch, history, dim)")
        batch, history, dim = match_embeds.shape
        if history != self.max_history or dim != self.embed_dim:
            raise ValueError(
                f"match_embeds must have shape (batch, {self.max_history}, "
                f"{self.embed_dim})"
            )
        if not isinstance(valid, torch.Tensor):
            raise TypeError("valid must be a torch.Tensor")
        if valid.shape != (batch, history):
            raise ValueError(f"valid must have shape ({batch}, {history})")
        valid = valid.to(dtype=torch.bool)

        has_any = valid.any(dim=1)
        valid_safe = valid.clone()
        if (~has_any).any():
            valid_safe[~has_any, 0] = True

        mask = self.log_time_decay[None, :].expand(batch, history)
        mask = mask + (~valid_safe).to(match_embeds.dtype) * NEG_INF
        attn_mask = (
            mask.unsqueeze(1)
            .expand(batch, self.num_heads, history)
            .reshape(batch * self.num_heads, 1, history)
        )
        query = self.match_query.expand(batch, 1, self.embed_dim)
        pooled, weights = self.match_attention(
            query,
            match_embeds,
            match_embeds,
            attn_mask=attn_mask,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        pooled = pooled.squeeze(1)
        pooled = pooled * has_any.to(pooled.dtype).unsqueeze(-1)
        return pooled, weights

    def encode_history(self, own_history, opp_history):
        """只编码历史，返回 `(batch, embed_dim)` 的池化嵌入。"""
        match_embeds, valid = self.encode_matches(own_history, opp_history)
        pooled, _ = self.pool_matches(match_embeds, valid)
        return pooled

    def encode_current_match(self, my_decisions, opp_decisions, rs, length, history_embed):
        """用 GRU 编码本场，并以历史嵌入/轮次为条件。"""
        if not isinstance(history_embed, torch.Tensor):
            raise TypeError("history_embed must be a torch.Tensor")
        if history_embed.ndim != 2 or history_embed.shape[1] != self.embed_dim:
            raise ValueError(f"history_embed must have shape (batch, {self.embed_dim})")
        batch = history_embed.shape[0]
        length = self._validate_current(my_decisions, opp_decisions, batch, length)
        round_ids = self._round_ids(rs, batch, history_embed.device)
        round_vec = self.round_embed(round_ids)
        hidden = torch.tanh(self.history_proj(history_embed) + round_vec).unsqueeze(0)

        if length == 0:
            current = hidden.squeeze(0)
        else:
            my = my_decisions[:, :length].to(history_embed.device)
            opp = opp_decisions[:, :length].to(history_embed.device)
            phase = self.phase_of_turn[:length]
            indices = my * 9 + opp * 3 + phase[None, :]
            step_embed = self.embed(indices) + round_vec.unsqueeze(1)
            output, _ = self.gru(step_embed, hidden)
            current = output[:, -1, :]
        return self.out_norm(current + history_embed)

    def forward(self, own_history, opp_history, my_decisions, opp_decisions, rs, length):
        """完整前向：历史 -> 池化 -> 本场 GRU -> 状态嵌入。"""
        own_history, opp_history = self._validate_history(own_history, opp_history)
        batch = own_history.shape[0]
        self._validate_current(my_decisions, opp_decisions, batch, length)
        match_embeds, valid = self.encode_matches(own_history, opp_history)
        history_embed, _ = self.pool_matches(match_embeds, valid)
        return self.encode_current_match(
            my_decisions, opp_decisions, rs, length, history_embed
        )


__all__ = [
    "HistoryEncoder",
    "INDEX_TO_MOVE",
    "MOVE_TO_INDEX",
    "NEXT_MOVE_TARGET",
    "VOCAB_SIZE",
    "moves_to_indices",
    "observation_tensors",
]
