"""辅助损失（issue #5）：预测对手下一手 + 对比学习。

这两项辅助任务只使用模拟器产生的客观标签，不引入人类先手先验：

- **预测对手下一手**：用当前状态嵌入预测对手下一步是合作还是背叛；
  标签来自模拟器记录，交叉熵损失。
- **对比学习**：同一策略的两场历史（两个视图）的池化嵌入应相近，
  不同策略的嵌入应分离；用对称 InfoNCE。

`NextMovePredictor` 是一个小 MLP，可接在 `HistoryEncoder` 输出的状态嵌入上；
`history_contrastive_loss` 直接复用编码器的 `encode_history()`。

设计给 #6 的策略网络使用；本模块只提供损失与头，不持有训练循环。
"""

import torch
from torch import nn
import torch.nn.functional as F


class NextMovePredictor(nn.Module):
    """从状态嵌入预测对手下一手 `(合作, 背叛)` 的 logits。"""

    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        hidden_dim = embed_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, state_embed):
        return self.net(state_embed)


def next_move_loss(logits, target):
    """对手下一手预测的交叉熵。

    Args:
        logits: `(batch, 2)`，第 0 列合作、第 1 列背叛。
        target: `(batch,)`，0 表示合作、1 表示背叛。

    Returns:
        标量损失。
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("logits must have shape (batch, 2)")
    if logits.shape[1] != 2:
        raise ValueError("logits must have 2 classes")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    target = target.long()
    if target.shape != (logits.shape[0],):
        raise ValueError("target must have shape (batch,)")
    return F.cross_entropy(logits, target)


def contrastive_loss(anchor, positive, temperature=0.1):
    """对称 InfoNCE：anchor[i] 与 positive[i] 为正样本，其余为负样本。

    Args:
        anchor: `(batch, dim)`。
        positive: `(batch, dim)`，与 anchor 一一对应。
        temperature: 温度系数，必须为正。

    Returns:
        标量损失；`batch < 2` 时没有负样本，返回与输入相连的 0。
    """
    if not isinstance(anchor, torch.Tensor) or not isinstance(positive, torch.Tensor):
        raise TypeError("anchor and positive must be torch.Tensor")
    if anchor.ndim != 2 or positive.ndim != 2:
        raise ValueError("anchor and positive must have shape (batch, dim)")
    if anchor.shape != positive.shape:
        raise ValueError("anchor and positive must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if anchor.shape[0] < 2:
        # 保留计算图，便于调用方统一 backward。
        return anchor.sum() * 0.0 + positive.sum() * 0.0

    anchor = F.normalize(anchor, dim=-1, eps=1e-8)
    positive = F.normalize(positive, dim=-1, eps=1e-8)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )


def history_contrastive_loss(encoder, view_a, view_b, temperature=0.1):
    """同一策略两场历史（两个视图）的对比损失。

    Args:
        encoder: 具备 `encode_history(own_history, opp_history)` 的编码器。
        view_a/view_b: 各自 `(own_history, opp_history)` 张量二元组。
    """
    own_a, opp_a = view_a
    own_b, opp_b = view_b
    embed_a = encoder.encode_history(own_a, opp_a)
    embed_b = encoder.encode_history(own_b, opp_b)
    return contrastive_loss(embed_a, embed_b, temperature=temperature)


def combined_auxiliary_loss(
    next_logits,
    next_target,
    anchor,
    positive,
    next_weight=1.0,
    contrastive_weight=0.1,
    temperature=0.1,
):
    """把两项辅助损失按权重相加，供 #6 直接调用。"""
    if next_weight < 0 or contrastive_weight < 0:
        raise ValueError("loss weights must be non-negative")
    return (
        next_weight * next_move_loss(next_logits, next_target)
        + contrastive_weight * contrastive_loss(anchor, positive, temperature=temperature)
    )


__all__ = [
    "NextMovePredictor",
    "combined_auxiliary_loss",
    "contrastive_loss",
    "history_contrastive_loss",
    "next_move_loss",
]
