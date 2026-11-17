"""阶段调度与验收（issue #6，sub-issue 8.4；ADR 0006）。

四个阶段按 ADR 0006 引入对手复杂度与历史隐藏：

| 阶段 | 对手层 | hide_prob | 目标 |
|---:|---|---:|---|
| 1 | L0 + L1 | 0.0 | 基础对弈，超过基线 |
| 2 | L0–L2 | 0.0 | 固定规则读历史，regret 不增 |
| 3 | L0–L3 | 0.1 | 智能读历史 + mask（关键关口） |
| 4 | L0–L3 + 自我对弈 | 0.1 | 种群自我对弈收敛 |

本模块只定义阶段配置与验收逻辑，不导入 `rl_train`（避免循环依赖）；
训练循环用 `stage_setup()` 取对手池和配置，用 `acceptance_report()` 判定。

关于 issue 中的 "regret"：这里操作化为**相对基线的分差**
`regret(opponent) = baseline 对它的每场平均分 − 策略对它的每场平均分`。
正值表示策略在该对手上不如基线。基线是 issue #3 的参照尺（ADR 0001）。

阶段 1 的验收口径在 2026-11-17 做了**阶梯式放宽**（ADR 0007 修订）：
基线对 TFT 已到理论上限 72、对 AllC 是 118/120，要求学习策略在阶段 1
逐项 ≥ 基线几乎不可达；阶段 1 改为"防退化/诊断"关口，阶段 2/3 再收紧：

| 阶段 | 均值容差（可低于基线） | canonical 逐项 gap 上限 | 其他 |
|---|---:|---:|---|
| 1 | 3.0 | 15.0 | 记录 gap，不作为硬 ≤0 |
| 2 | 1.0 | 8.0 | 相对阶段 1 回退 ≤1.0 |
| 3 | 0（必须 > 基线） | 5.0 | hide 掉分 ≤15% |
| 4 | 0（必须 > 基线） | 5.0 | 自我对弈收敛 |
"""

from dataclasses import dataclass

from ipd.pool.registry import (
    STAGE_HIDE_PROB,
    STAGE_LAYERS,
    agents_for_stage,
    config_for_stage,
)

CANONICAL_OPPONENTS = ("all_c", "all_d", "tft", "gradual")

STAGE_DESCRIPTIONS = {
    1: "L0+L1 基础对弈（不读历史）",
    2: "加入 L2 固定规则读历史对手",
    3: "加入 L3 智能读历史对手 + mask（关键关口）",
    4: "种群自我对弈为主 + 保留 L0–L3",
}


@dataclass(frozen=True)
class StageSpec:
    """一个训练阶段的静态配置。"""

    stage: int
    layers: tuple
    hide_prob: float
    hide_scope: str
    iterations: int
    eval_every: int
    description: str
    self_play: bool = False
    # 阶段 3 的 hide_prob 鲁棒性门槛：隐藏历史导致的每场分数下降比例上限。
    hide_drop_limit: float = 0.15
    # 均值容差：策略允许低于基线的每场分数（阶段 1/2 阶梯放宽）。
    mean_tolerance: float = 0.5
    # canonical 逐项 gap 的上限（阶段 1 宽、阶段 3 收紧）。
    canonical_tolerance: float = 0.0
    # 阶段 2 相对阶段 1 的 regret 回退容差。
    regret_tolerance: float = 1.0


STAGE_SPECS = {
    1: StageSpec(
        stage=1,
        layers=STAGE_LAYERS[1],
        hide_prob=STAGE_HIDE_PROB[1],
        hide_scope="match",
        iterations=80,
        eval_every=20,
        description=STAGE_DESCRIPTIONS[1],
        mean_tolerance=3.0,
        canonical_tolerance=15.0,
    ),
    2: StageSpec(
        stage=2,
        layers=STAGE_LAYERS[2],
        hide_prob=STAGE_HIDE_PROB[2],
        hide_scope="match",
        iterations=80,
        eval_every=20,
        description=STAGE_DESCRIPTIONS[2],
        mean_tolerance=1.0,
        canonical_tolerance=8.0,
        regret_tolerance=1.0,
    ),
    3: StageSpec(
        stage=3,
        layers=STAGE_LAYERS[3],
        hide_prob=STAGE_HIDE_PROB[3],
        hide_scope="match",
        iterations=120,
        eval_every=20,
        description=STAGE_DESCRIPTIONS[3],
        canonical_tolerance=5.0,
    ),
    4: StageSpec(
        stage=4,
        layers=STAGE_LAYERS[4],
        hide_prob=STAGE_HIDE_PROB[4],
        hide_scope="match",
        iterations=150,
        eval_every=25,
        description=STAGE_DESCRIPTIONS[4],
        self_play=True,
        canonical_tolerance=5.0,
    ),
}


def stage_spec(stage):
    """返回阶段配置。

    Raises:
        ValueError: stage 不在 1–4。
    """
    try:
        return STAGE_SPECS[stage]
    except KeyError:
        raise ValueError(f"Unknown stage: {stage}") from None


def stage_setup(stage, seed=0, variants=None):
    """按 ADR 0006 构造阶段对手池与评测配置。

    Args:
        stage: 1–4。
        seed: 随机参数变体的种子。
        variants: 每个可随机族的变体数；None 用池默认值。

    Returns:
        `(StageSpec, opponents, TournamentConfig)`。
    """
    spec = stage_spec(stage)
    opponents = agents_for_stage(stage, seed=seed, variants=variants)
    config = config_for_stage(stage)
    if spec.layers != STAGE_LAYERS[stage]:
        raise AssertionError("stage layers out of sync with registry")
    if abs(spec.hide_prob - config.hide_prob) > 1e-9:
        raise AssertionError("stage hide_prob out of sync with registry")
    return spec, opponents, config


def stage_checkpoint_dir(root, stage):
    """阶段检查点目录（`<root>/stage<N>`）。"""
    from pathlib import Path

    stage_spec(stage)
    return Path(root) / f"stage{stage}"


@dataclass(frozen=True)
class GateResult:
    """阶段验收结果。`reasons` 是硬性失败原因；`notes` 是未测量/软性提示。"""

    stage: int
    passed: bool
    reasons: tuple
    notes: tuple = ()

    def __bool__(self):
        return self.passed


def _canonical_regrets(metrics):
    """取出四个 canonical 对手的 regret；缺失的记入 notes。"""
    regret = metrics.get("regret") or {}
    values, missing = {}, []
    for name in CANONICAL_OPPONENTS:
        if name in regret:
            values[name] = float(regret[name])
        else:
            missing.append(name)
    return values, missing


def acceptance_report(stage, metrics, previous=None):
    """按阶段验收条件判断训练结果。

    Args:
        stage: 1–4。
        metrics: `evaluate_policy()` 的结果；阶段 3/4 可含
            `hide_drop_ratio`，阶段 4 可含 `converged`。
        previous: 上一阶段的 metrics（阶段 2 用来比较 regret 是否回退）。

    Returns:
        `GateResult`。

    Raises:
        ValueError: stage 非法或缺少策略/基线分数。
    """
    spec = stage_spec(stage)
    if "policy_per_match" not in metrics:
        raise ValueError("metrics missing policy_per_match")
    policy_score = float(metrics["policy_per_match"])
    baseline_score = metrics.get("baseline_per_match")
    if baseline_score is None:
        raise ValueError("metrics missing baseline_per_match")
    baseline_score = float(baseline_score)
    reasons, notes = [], []
    regrets, missing = _canonical_regrets(metrics)
    if missing:
        notes.append(f"缺少 canonical 对手: {', '.join(missing)}")

    if stage == 1:
        if policy_score < baseline_score - spec.mean_tolerance:
            reasons.append(
                f"阶段 1 平均分低于基线超过 {spec.mean_tolerance}: "
                f"{policy_score:.2f} vs {baseline_score:.2f}"
            )
        for name, value in regrets.items():
            if value > spec.canonical_tolerance:
                reasons.append(
                    f"对 {name} 的 regret={value:.2f} > "
                    f"{spec.canonical_tolerance:.1f}"
                )
            elif value > 0:
                notes.append(
                    f"对 {name} 的 regret={value:.2f} 在阶段 1 容差 "
                    f"{spec.canonical_tolerance:.1f} 内"
                )
    elif stage == 2:
        if policy_score < baseline_score - spec.mean_tolerance:
            reasons.append(
                f"阶段 2 平均分低于基线超过 {spec.mean_tolerance}: "
                f"{policy_score:.2f} vs {baseline_score:.2f}"
            )
        for name, value in regrets.items():
            if value > spec.canonical_tolerance:
                reasons.append(
                    f"阶段 2 对 {name} 的 regret={value:.2f} > "
                    f"{spec.canonical_tolerance:.1f}"
                )
        if previous is not None:
            previous_regrets, _ = _canonical_regrets(previous)
            for name, value in regrets.items():
                if name in previous_regrets:
                    allowed = previous_regrets[name] + spec.regret_tolerance
                    if value > allowed:
                        reasons.append(
                            f"对 {name} 的 regret 回退: {value:.2f} > "
                            f"{previous_regrets[name]:.2f}+{spec.regret_tolerance}"
                        )
    elif stage == 3:
        if not policy_score > baseline_score:
            reasons.append(
                f"阶段 3 平均分未超过基线: {policy_score:.2f} <= {baseline_score:.2f}"
            )
        for name, value in regrets.items():
            if value > spec.canonical_tolerance:
                reasons.append(
                    f"阶段 3 对 {name} 的 regret={value:.2f} > "
                    f"{spec.canonical_tolerance:.1f}"
                )
        ratio = metrics.get("hide_drop_ratio")
        if ratio is None:
            notes.append("未测量 hide_prob 鲁棒性（hide_drop_ratio）")
        elif float(ratio) > spec.hide_drop_limit:
            reasons.append(
                f"隐藏历史导致分数下降 {float(ratio):.1%}，"
                f"超过 {spec.hide_drop_limit:.0%}"
            )
        if not metrics.get("learned_reputation", True):
            notes.append("未验证是否学到防范读历史对手（需行为差异检查）")
    else:
        for name, value in regrets.items():
            if value > spec.canonical_tolerance:
                reasons.append(
                    f"阶段 4 对 {name} 的 regret={value:.2f} > "
                    f"{spec.canonical_tolerance:.1f}"
                )
        if not metrics.get("converged", False):
            reasons.append("阶段 4 种群自我对弈尚未收敛")
        if "hide_drop_ratio" not in metrics:
            notes.append("未测量 hide_prob 鲁棒性（hide_drop_ratio）")

    return GateResult(
        stage=stage,
        passed=not reasons,
        reasons=tuple(reasons),
        notes=tuple(notes),
    )


__all__ = [
    "CANONICAL_OPPONENTS",
    "GateResult",
    "STAGE_SPECS",
    "StageSpec",
    "acceptance_report",
    "stage_checkpoint_dir",
    "stage_setup",
    "stage_spec",
]
