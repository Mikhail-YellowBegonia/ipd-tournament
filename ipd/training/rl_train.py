"""强化学习训练循环（issue #6，sub-issue 8.3/8.5/8.6）。

算法：REINFORCE + 价值基线（advantage = 回报 − V(s)）+ 熵正则，外加 issue #5
提供的两项辅助损失（预测对手下一手、历史对比学习）。策略分布只在动作头上，
编码器与价值头共享状态嵌入。每个训练赛季（8 轮 × 每轮若干对手）采一批轨迹，
按回合长度分组后做一次小批量梯度更新。

设计取舍：

- **按长度分组**：编码器要求同一 batch 的本场动作长度一致；同一轮所有对局
  同步推进，所以长度天然成组，不需要 padding。
- **价值基线**：纯 REINFORCE 的方差在 25 回合、收益分三段的设定下太大；
  价值头用同一条轨迹的回报做 MSE 回归，只做基线不做自举（γ=1 的有限期）。
- **辅助损失**：只使用模拟器标签；下一手预测对每个长度 < 24 的转移都有标签，
  对比学习在同一场历史的两个随机掩码视图之间进行。
- **评估**：`evaluate_policy()` 把策略包装成确定性 Agent，与对手池和基线跑
  完整 8 轮锦标赛，报告每场平均分、名次、逐对手分数与相对基线的 regret。

训练不受 0.5 s 决策限制；训练环境的 `TournamentConfig.timeout` 不生效。
"""

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from ipd.agents.encoder import NEXT_MOVE_TARGET
from ipd.pool.registry import agents_for_stage, config_for_stage
from ipd.sim import run_tournament
from ipd.training.auxiliary import (
    NextMovePredictor,
    history_contrastive_loss,
    next_move_loss,
)
from ipd.training.env import TrainingEnv
from ipd.training.policy import IPDPolicy, policy_agent
from ipd.training.schedule import acceptance_report, stage_setup


@dataclass
class TrainConfig:
    """训练超参数（与阶段无关的部分）。"""

    lr: float = 1e-3
    gamma: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    next_move_coef: float = 0.1
    contrastive_coef: float = 0.05
    contrastive_temperature: float = 0.1
    contrastive_matches: int = 16
    max_grad_norm: float = 1.0
    batch_size: int = 256
    minibatch_epochs: int = 1
    opponents_per_round: int = 30
    shaping: float = 0.0
    seed: int = 0
    threads: int = 1
    eval_seeds: tuple = (0,)
    eval_robustness: bool = False

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        for name in (
            "entropy_coef",
            "value_coef",
            "next_move_coef",
            "contrastive_coef",
            "shaping",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.minibatch_epochs <= 0:
            raise ValueError("minibatch_epochs must be positive")
        if self.contrastive_matches < 0:
            raise ValueError("contrastive_matches must be non-negative")
        if not self.eval_seeds:
            raise ValueError("eval_seeds must not be empty")


def compute_returns(rewards, gamma=1.0):
    """有限期折扣回报 `G_t = sum_{k>=t} gamma^(k-t) r_k`。"""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def _group_transitions(episode, gamma):
    """把轨道按回合长度分组，并计算归一化回报与优势。

    回报整体按标准差缩放，让价值损失与策略损失同量级；优势再做一次
    零均值/单位方差归一化，降低不同赛季之间的梯度尺度差异。
    """
    groups = {}
    raw_returns = []
    for match in episode.matches:
        returns = compute_returns(match.rewards, gamma)
        for length in range(len(match.actions)):
            groups.setdefault(length, []).append((match, length, returns[length]))
            raw_returns.append(returns[length])
    if not raw_returns:
        raise ValueError("episode has no transitions")
    return_mean = sum(raw_returns) / len(raw_returns)
    return_var = sum((value - return_mean) ** 2 for value in raw_returns) / len(raw_returns)
    return_std = return_var ** 0.5 or 1.0

    normalized = {}
    raw_advantages = []
    for length, items in groups.items():
        rows = []
        for match, t, returns in items:
            normalized_return = (returns - return_mean) / return_std
            normalized_value = (match.values[t] - return_mean) / return_std
            rows.append((match, t, normalized_return))
            raw_advantages.append(normalized_return - normalized_value)
        normalized[length] = rows
    adv_mean = sum(raw_advantages) / len(raw_advantages)
    adv_var = sum((value - adv_mean) ** 2 for value in raw_advantages) / len(raw_advantages)
    adv_std = adv_var ** 0.5 or 1.0

    result = {}
    for length, rows in normalized.items():
        result[length] = []
        for match, t, normalized_return in rows:
            normalized_value = (match.values[t] - return_mean) / return_std
            advantage = (
                normalized_return - normalized_value - adv_mean
            ) / adv_std
            result[length].append((match, t, normalized_return, advantage))
    return result


def _iter_batches(groups, batch_size, generator=None):
    """按长度遍历小批量；长度内部打乱。"""
    for length in sorted(groups):
        items = list(groups[length])
        if generator is None:
            order = list(range(len(items)))
        else:
            order = torch.randperm(len(items), generator=generator).tolist()
        for start in range(0, len(items), batch_size):
            yield length, [items[i] for i in order[start:start + batch_size]]


def _batch_tensors(items, device=None):
    """把同一长度的转移组装成编码器输入与训练标签。"""
    length = items[0][1]
    own = torch.stack([match.own_history for match, _, _, _ in items]).to(device)
    opp = torch.stack([match.opp_history for match, _, _, _ in items]).to(device)
    my = torch.tensor(
        [[action + 1 for action in match.actions[:length]] for match, _, _, _ in items],
        dtype=torch.long,
        device=device,
    )
    opp_moves = torch.tensor(
        [
            [1 if move == "c" else 2 for move in match.opp_moves[:length]]
            for match, _, _, _ in items
        ],
        dtype=torch.long,
        device=device,
    )
    rs = torch.tensor([match.rs for match, _, _, _ in items], dtype=torch.long, device=device)
    actions = torch.tensor(
        [match.actions[length] for match, _, _, _ in items], dtype=torch.long, device=device
    )
    returns = torch.tensor(
        [returns for _, _, returns, _ in items], dtype=torch.float32, device=device
    )
    advantages = torch.tensor(
        [advantage for _, _, _, advantage in items], dtype=torch.float32, device=device
    )
    next_mask = length < 24
    if next_mask:
        next_targets = torch.tensor(
            [NEXT_MOVE_TARGET[match.opp_moves[length + 1]] for match, _, _, _ in items],
            dtype=torch.long,
            device=device,
        )
        next_flags = torch.ones(len(items), dtype=torch.bool, device=device)
    else:
        next_targets = torch.zeros(len(items), dtype=torch.long, device=device)
        next_flags = torch.zeros(len(items), dtype=torch.bool, device=device)
    return {
        "length": length,
        "own_history": own,
        "opp_history": opp,
        "my_decisions": my,
        "opp_decisions": opp_moves,
        "rs": rs,
        "actions": actions,
        "returns": returns,
        "advantages": advantages,
        "next_targets": next_targets,
        "next_mask": next_flags,
        "old_log_probs": torch.tensor(
            [match.log_probs[length] for match, _, _, _ in items],
            dtype=torch.float32,
            device=device,
        ),
    }


def _contrastive_views(matches, generator, hidden_prob=0.1, limit=16):
    """同一场历史的两个随机掩码视图（掩码同步作用于两份历史）。"""
    matches = list(matches)
    if len(matches) > limit:
        order = torch.randperm(len(matches), generator=generator)[:limit].tolist()
        matches = [matches[i] for i in order]
    selected = matches
    if len(selected) < 2:
        return None
    own = torch.stack([match.own_history for match in selected])
    opp = torch.stack([match.opp_history for match in selected])
    views = []
    for _ in range(2):
        mask = torch.rand(own.shape, generator=generator) < hidden_prob
        views.append((own.masked_fill(mask, 0), opp.masked_fill(mask, 0)))
    return views


def update_policy(policy, predictor, optimizer, episode, config, generator=None):
    """用一批轨迹做一次策略梯度更新，返回指标字典。"""
    groups = _group_transitions(episode, config.gamma)
    batches = list(_iter_batches(groups, config.batch_size, generator))
    if not batches:
        raise ValueError("no training batches")
    device = next(policy.parameters()).device
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "next_move_loss": 0.0,
        "contrastive_loss": 0.0,
        "transitions": 0,
    }
    contrastive_views = None
    if config.contrastive_coef > 0:
        contrastive_views = _contrastive_views(
            episode.matches, generator, limit=config.contrastive_matches
        )

    last_grad_norm = 0.0
    for _epoch in range(config.minibatch_epochs):
        optimizer.zero_grad()
        for _length, items in batches:
            batch = _batch_tensors(items, device=device)
            state = policy.encode(
                batch["own_history"],
                batch["opp_history"],
                batch["my_decisions"],
                batch["opp_decisions"],
                batch["rs"],
                batch["length"],
            )
            logits = policy.action_head(state)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probs = distribution.log_prob(batch["actions"])
            entropy = distribution.entropy()
            policy_loss = -(log_probs * batch["advantages"]).mean()
            loss = policy_loss - config.entropy_coef * entropy.mean()
            totals["policy_loss"] += float(policy_loss.item())
            totals["entropy"] += float(entropy.mean().item())

            if policy.use_value:
                values = policy.value_head(state).squeeze(-1)
                value_loss = F.mse_loss(values, batch["returns"])
                loss = loss + config.value_coef * value_loss
                totals["value_loss"] += float(value_loss.item())

            if predictor is not None and config.next_move_coef > 0:
                mask = batch["next_mask"]
                if bool(mask.any()):
                    next_logits = predictor(state)
                    next_loss = next_move_loss(
                        next_logits[mask], batch["next_targets"][mask]
                    )
                    loss = loss + config.next_move_coef * next_loss
                    totals["next_move_loss"] += float(next_loss.item())

            totals["loss"] += float(loss.item())
            totals["transitions"] += len(items)
            (loss / len(batches)).backward()

        if contrastive_views is not None:
            loss_c = history_contrastive_loss(
                policy.encoder,
                contrastive_views[0],
                contrastive_views[1],
                temperature=config.contrastive_temperature,
            )
            (config.contrastive_coef * loss_c / config.minibatch_epochs).backward()
            totals["contrastive_loss"] += float(loss_c.item())

        last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters())
                + (list(predictor.parameters()) if predictor is not None else []),
                config.max_grad_norm,
            )
        )
        optimizer.step()

    batch_divisor = max(config.minibatch_epochs * len(batches), 1)
    epoch_divisor = max(config.minibatch_epochs, 1)
    metrics = {
        key: totals[key] / batch_divisor
        for key in ("loss", "policy_loss", "value_loss", "entropy", "next_move_loss")
    }
    metrics["contrastive_loss"] = totals["contrastive_loss"] / epoch_divisor
    metrics["transitions"] = totals["transitions"] / epoch_divisor
    metrics["grad_norm"] = last_grad_norm
    metrics["train_score"] = episode.mean_score
    return metrics


def evaluate_policy(
    policy,
    opponents,
    config,
    seeds=(0,),
    baseline=None,
    canonical=("all_c", "all_d", "tft", "gradual"),
    include_hide_robustness=False,
):
    """让策略以确定性 Agent 身份与对手池 + 基线跑完整锦标赛。

    Args:
        policy: `IPDPolicy`。
        opponents: 对手 Agent 列表。
        config: `TournamentConfig`（评测用，含 hide_prob）。
        seeds: 评估种子；每个种子跑一次完整锦标赛。
        baseline: 可选的基线 Agent，用于逐对手 regret。
        canonical: 需要单独报告 regret 的标准对手名。

    Returns:
        指标字典（JSON 友好）：
        `policy_per_match`、`baseline_per_match`、`rank`、`faults`、
        `per_opponent`、`regret`、`seeds`。
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("need at least one evaluation seed")
    opponents = list(opponents)
    names = {agent.name for agent in opponents}
    if baseline is not None:
        names.add(baseline.name)
    candidate_name = "policy"
    suffix = 1
    while candidate_name in names:
        suffix += 1
        candidate_name = f"policy#{suffix}"
    candidate = policy_agent(policy, name=candidate_name)
    players = [candidate] + opponents + ([baseline] if baseline is not None else [])
    totals = {agent.name: 0.0 for agent in players}
    pair_totals = {agent.name: {other.name: 0.0 for other in players} for agent in players}
    faults = 0
    for seed in seeds:
        result = run_tournament(players, seed=seed, cfg=config)
        faults += sum(result.faults)
        for name, score in zip(result.names, result.scores):
            totals[name] += score
        for i, row in enumerate(result.names):
            for j, other in enumerate(result.names):
                pair_totals[row][other] += result.pair_scores[i][j]
    opponent_count = len(players) - 1
    per_match_divisor = config.rounds * opponent_count

    def per_match(name):
        return totals[name] / (len(seeds) * per_match_divisor)

    per_opponent = {}
    for opponent in opponents:
        per_opponent[opponent.name] = (
            pair_totals[candidate_name][opponent.name] / (len(seeds) * config.rounds)
        )
    regret = {}
    baseline_per_match = None
    if baseline is not None:
        baseline_per_match = per_match(baseline.name)
        for opponent in opponents:
            if opponent.name in canonical:
                baseline_vs = (
                    pair_totals[baseline.name][opponent.name]
                    / (len(seeds) * config.rounds)
                )
                policy_vs = per_opponent[opponent.name]
                regret[opponent.name] = baseline_vs - policy_vs

    # 名次按总得分（与模拟器排名一致）。
    ranking = sorted(
        ((name, totals[name]) for name in totals), key=lambda item: -item[1]
    )
    rank = 1 + sum(1 for _, score in ranking if score > totals[candidate_name])
    result = {
        "policy_per_match": per_match(candidate_name),
        "baseline_per_match": baseline_per_match,
        "rank": rank,
        "n_agents": len(players),
        "faults": faults,
        "per_opponent": per_opponent,
        "regret": regret,
        "seeds": list(seeds),
        "total_policy": totals[candidate_name],
    }
    if include_hide_robustness and config.hide_prob > 0:
        clear = evaluate_policy(
            policy,
            opponents,
            replace(config, hide_prob=0.0),
            seeds=seeds,
            baseline=baseline,
            canonical=canonical,
            include_hide_robustness=False,
        )
        result["clear_per_match"] = clear["policy_per_match"]
        result["hide_drop"] = clear["policy_per_match"] - result["policy_per_match"]
        result["hide_drop_ratio"] = result["hide_drop"] / max(
            abs(clear["policy_per_match"]), 1e-9
        )
    return result


def _checkpoint_payload(policy, predictor, optimizer, train_config, stage, iteration, extra=None):
    return {
        "stage": stage,
        "iteration": iteration,
        "policy_config": policy.config(),
        "policy_state": policy.state_dict(),
        "predictor_state": predictor.state_dict() if predictor is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "train_config": asdict(train_config),
        "extra": extra or {},
    }


def save_checkpoint(path, payload):
    """保存检查点（torch.save）；返回 Path。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path, policy=None):
    """加载检查点；给定 policy 时把权重写回并返回 (policy, payload)。"""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if policy is None:
        policy = IPDPolicy.from_config(payload["policy_config"])
    policy.load_state_dict(payload["policy_state"])
    return policy, payload


def train_stage(
    stage,
    policy,
    opponents,
    config,
    baseline=None,
    num_episodes=100,
    train_config=None,
    checkpoint_dir=None,
    eval_every=None,
    log_path=None,
    verbose=True,
):
    """训练一个阶段（issue 接口 + 训练超参数）。

    Args:
        stage: 阶段编号 1–4（仅用于日志/检查点）。
        policy: `IPDPolicy`。
        opponents: 该阶段对手池（`agents_for_stage`）。
        config: `TournamentConfig`（阶段配置，评测时使用）。
        baseline: 可选基线 Agent，用于定期评估。
        num_episodes: 训练赛季数（issue 里叫 episodes）。
        train_config: `TrainConfig`；None 用默认值。
        checkpoint_dir: 检查点目录；None 不保存。
        eval_every: 每多少个赛季评估一次；None 或 0 表示不评估。
        log_path: JSONL 指标日志路径；None 不写。
        verbose: 是否打印进度。

    Returns:
        指标历史列表（每项含 iteration、训练指标、可选 eval）。
    """
    train_config = train_config or TrainConfig()
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    env = TrainingEnv(
        opponents,
        cfg=config,
        seed=train_config.seed,
        opponents_per_round=train_config.opponents_per_round,
    )
    predictor = (
        NextMovePredictor(policy.embed_dim)
        if train_config.next_move_coef > 0
        else None
    )
    parameters = list(policy.parameters()) + (
        list(predictor.parameters()) if predictor is not None else []
    )
    optimizer = torch.optim.Adam(parameters, lr=train_config.lr)
    history = []
    best_score = None
    log_file = Path(log_path) if log_path else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("", encoding="utf-8")

    for iteration in range(num_episodes):
        env.reset(seed=train_config.seed + iteration)
        generator = torch.Generator().manual_seed(
            train_config.seed * 1_000_003 + iteration
        )
        policy.eval()
        episode = env.rollout(
            policy, shaping=train_config.shaping, generator=generator
        )
        policy.train()
        metrics = update_policy(
            policy, predictor, optimizer, episode, train_config, generator=generator
        )
        metrics["iteration"] = iteration
        metrics["stage"] = stage
        eval_result = None
        if eval_every and (iteration + 1) % eval_every == 0:
            policy.eval()
            eval_result = evaluate_policy(
                policy,
                opponents,
                config,
                seeds=train_config.eval_seeds,
                baseline=baseline,
                include_hide_robustness=(
                    stage >= 3 and train_config.eval_robustness
                ),
            )
            metrics["eval"] = eval_result
            if baseline is not None:
                gate = acceptance_report(stage, eval_result)
                metrics["gate"] = {"passed": gate.passed, "reasons": list(gate.reasons)}
            score = eval_result["policy_per_match"]
            if best_score is None or score > best_score:
                best_score = score
                if checkpoint_dir:
                    save_checkpoint(
                        Path(checkpoint_dir) / f"stage{stage}_best.pt",
                        _checkpoint_payload(
                            policy,
                            predictor,
                            optimizer,
                            train_config,
                            stage,
                            iteration,
                            extra={"eval": eval_result},
                        ),
                    )
        if checkpoint_dir:
            save_checkpoint(
                Path(checkpoint_dir) / f"stage{stage}_last.pt",
                _checkpoint_payload(
                    policy,
                    predictor,
                    optimizer,
                    train_config,
                    stage,
                    iteration,
                    extra={"metrics": metrics},
                ),
            )
        history.append(metrics)
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if verbose:
            line = (
                f"[stage {stage}] iter {iteration + 1}/{num_episodes} "
                f"train {metrics['train_score']:.2f} "
                f"loss {metrics['loss']:.3f} "
                f"entropy {metrics['entropy']:.3f}"
            )
            if eval_result is not None:
                line += (
                    f" | eval {eval_result['policy_per_match']:.2f} "
                    f"(baseline {eval_result['baseline_per_match']:.2f})"
                    if eval_result["baseline_per_match"] is not None
                    else f" | eval {eval_result['policy_per_match']:.2f}"
                )
            print(line)
    return history


def _parse_seeds(text):
    return tuple(int(part) for part in text.split(",") if part.strip())


def build_parser():
    parser = argparse.ArgumentParser(description="按 ADR 0006 分阶段训练 IPD 策略")
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--iterations", type=int, default=None, help="覆盖阶段默认赛季数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--opponents-per-round", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=None, help="0 表示只在训练后评估")
    parser.add_argument("--eval-seeds", default="0,1")
    parser.add_argument("--shaping", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", default="runs/rl")
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--variants", type=int, default=None)
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="阶段 3/4 评估时额外跑 hide_prob=0 的对照（耗时翻倍）",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    spec, opponents, config = stage_setup(
        args.stage, seed=args.seed, variants=args.variants
    )
    policy = IPDPolicy()
    if args.resume:
        policy, payload = load_checkpoint(args.resume, policy=policy)
        print(f"resumed from {args.resume} (iteration {payload.get('iteration')})")
    if args.no_baseline:
        baseline = None
    else:
        from ipd.agents.baseline import baseline_agent

        baseline = baseline_agent()
    train_config = TrainConfig(
        seed=args.seed,
        threads=args.threads,
        opponents_per_round=args.opponents_per_round,
        shaping=args.shaping,
        eval_seeds=_parse_seeds(args.eval_seeds),
        eval_robustness=args.robustness,
    )
    eval_every = spec.eval_every if args.eval_every is None else args.eval_every
    iterations = spec.iterations if args.iterations is None else args.iterations
    checkpoint_dir = Path(args.checkpoint_dir) / f"stage{args.stage}"
    print(
        f"stage {args.stage}: {len(opponents)} opponents, "
        f"hide_prob={config.hide_prob}, iterations={iterations}, "
        f"eval_every={eval_every}, checkpoint={checkpoint_dir}"
    )
    history = train_stage(
        args.stage,
        policy,
        opponents,
        config,
        baseline=baseline,
        num_episodes=iterations,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        eval_every=eval_every,
        log_path=args.log_path,
        verbose=True,
    )
    final = history[-1] if history else {}
    if "eval" in final:
        gate = acceptance_report(args.stage, final["eval"])
        print(f"final gate: {'PASS' if gate.passed else 'FAIL'} {list(gate.reasons)}")
    return history


if __name__ == "__main__":
    main()
