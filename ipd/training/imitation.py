"""行为克隆热启动（issue #6 的训练准备，与 #8 蒸馏共享实现）。

动机：纯 REINFORCE 从零训练时，策略会收敛到"首手背叛"的局部最优，
对 TFT/Grim/STFT 等报复型对手锁定互相背叛；100–200 赛季的评估仍只有
18–32 分/场，而手工基线约 50 分/场。这里先用基线作为**教师**做监督预训练，
把"首手善意、按对手行为调整、终局收割"这些行为注入策略网络，再用
RL 微调（`ipd.training.rl_train --resume`）。

数据流：

1. `BaselineTeacher` 把训练环境的批量张量解码成考试九参数，调用
   `ipd.agents.baseline.strategy` 得到动作；
2. `collect_demonstrations()` 用教师跑若干赛季，得到与 RL 环境完全同分布的
   状态-动作轨迹；
3. `train_behavior_cloning()` 用交叉熵训练编码器 + 动作头；
4. 检查点可被 `ipd.training.rl_train` 直接 `--resume`。

这不是把基线当最终产品：基线的动作只用于初始化，最终模型仍要经过 RL 微调，
并在留出池上按 ADR 0001 与基线比较。若 BC-only 已经接近/超过基线，则 RL
微调只做小幅增益，不会再从零探索。
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ipd.agents.baseline import strategy as baseline_strategy
from ipd.agents.encoder import INDEX_TO_MOVE
from ipd.sim import rules
from ipd.training.env import TrainingEnv
from ipd.training.policy import COOPERATE_INDEX, DEFECT_INDEX, IPDPolicy
from ipd.training.rl_train import (
    evaluate_policy,
    save_checkpoint,
)

DEFAULT_CHECKPOINT_DIR = "runs/rl"


def _decode_history(tensor):
    """把 (20, 25) 索引张量解码成考试接口的字符串行元组。"""
    return tuple(
        tuple(INDEX_TO_MOVE[int(value)] for value in row)
        for row in tensor.tolist()
    )


class BaselineTeacher:
    """把手工基线包装成训练环境可批量调用的"策略"。

    训练环境在每一轮把同批对手的历史拼接成 `(batch, 20, 25)` 张量；历史
    在整轮内不变，所以在 `length == 0` 时清空解码缓存，之后按张量地址复用
    字符串元组，避免每个回合重复解码 20×25 历史。
    """

    def __init__(self, strategy=baseline_strategy):
        if not callable(strategy):
            raise TypeError("strategy must be callable")
        self.strategy = strategy
        self._cache = {}

    def _history_rows(self, own_history, opp_history, index):
        key = (own_history.data_ptr(), opp_history.data_ptr(), index)
        cached = self._cache.get(key)
        if cached is None:
            cached = (
                _decode_history(own_history[index]),
                _decode_history(opp_history[index]),
            )
            self._cache[key] = cached
        return cached

    def act_with_value(
        self,
        own_history,
        opp_history,
        my_decisions,
        opp_decisions,
        rs,
        length,
        deterministic=False,
        generator=None,
    ):
        if int(length) == 0:
            self._cache.clear()
        batch = int(own_history.shape[0])
        my_rows = my_decisions.tolist()
        opp_rows = opp_decisions.tolist()
        round_ids = rs.tolist()
        actions = []
        for index in range(batch):
            own_rows, opponent_rows = self._history_rows(
                own_history, opp_history, index
            )
            try:
                move = self.strategy(
                    int(round_ids[index]),
                    int(length),
                    1,
                    _decode_move_row(my_rows[index]),
                    2,
                    _decode_move_row(opp_rows[index]),
                    rules.REPAY,
                    own_rows,
                    opponent_rows,
                )
            except Exception:
                move = rules.COOPERATE
            actions.append(COOPERATE_INDEX if move == rules.COOPERATE else DEFECT_INDEX)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        probs = torch.zeros((batch, 2), dtype=torch.float32)
        probs[torch.arange(batch), action_tensor] = 1.0
        return action_tensor, probs, torch.zeros(batch, dtype=torch.float32)


def _decode_move_row(row):
    """把一行动作索引解码成 `"c"`/`"b"` 元组。"""
    return tuple(INDEX_TO_MOVE[int(value)] for value in row)


def collect_demonstrations(
    opponents,
    config,
    seasons=20,
    seed=0,
    opponents_per_round=24,
    teacher=None,
    verbose=False,
):
    """用教师策略跑若干赛季，返回轨迹列表（与 RL 环境同分布）。"""
    if seasons <= 0:
        raise ValueError("seasons must be positive")
    teacher = teacher or BaselineTeacher()
    env = TrainingEnv(
        opponents,
        cfg=config,
        seed=seed,
        opponents_per_round=opponents_per_round,
    )
    episodes = []
    for season in range(seasons):
        env.reset(seed=seed + season)
        generator = torch.Generator().manual_seed(seed * 1_000_003 + season)
        episode = env.rollout(teacher, generator=generator)
        episodes.append(episode)
        if verbose:
            print(
                f"  demo season {season + 1}/{seasons}: "
                f"{len(episode.matches)} matches, "
                f"mean score {episode.mean_score:.2f}"
            )
    return episodes


def collect_student_rollouts(
    policy,
    opponents,
    config,
    seasons=10,
    seed=0,
    opponents_per_round=24,
    verbose=False,
):
    """用当前学生策略采样状态，供 DAgger 标注（动作按学生分布采样）。"""
    if seasons <= 0:
        raise ValueError("seasons must be positive")
    policy.eval()
    env = TrainingEnv(
        opponents,
        cfg=config,
        seed=seed,
        opponents_per_round=opponents_per_round,
    )
    episodes = []
    for season in range(seasons):
        env.reset(seed=seed + season)
        generator = torch.Generator().manual_seed(seed * 1_000_003 + season)
        episode = env.rollout(policy, generator=generator)
        episodes.append(episode)
        if verbose:
            print(
                f"  student season {season + 1}/{seasons}: "
                f"{len(episode.matches)} matches, "
                f"mean score {episode.mean_score:.2f}"
            )
    return episodes


def label_episodes_with_teacher(episodes, teacher=None, batch_size=256):
    """用教师给轨迹里的每个状态打动作标签（DAgger 的聚合步骤）。

    行为动作保留在 `match.actions`（学生采样），标签写入 `match.labels`；
    训练时用行为前缀构造 `my_decisions`，用教师标签算交叉熵。
    """
    teacher = teacher or BaselineTeacher()
    groups = _group_demonstrations(episodes)
    labeled = 0
    for length in sorted(groups):
        teacher._cache.clear()
        items = groups[length]
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            batch = _batch_tensors(chunk)
            actions, _, _ = teacher.act_with_value(
                batch["own_history"],
                batch["opp_history"],
                batch["my_decisions"],
                batch["opp_decisions"],
                batch["rs"],
                length,
            )
            for (match, _), action in zip(chunk, actions.tolist()):
                if len(match.labels) != len(match.actions):
                    match.labels = [None] * len(match.actions)
                match.labels[length] = int(action)
                labeled += 1
    return labeled


def _group_demonstrations(episodes):
    """按回合长度分组，返回 `{length: [(match, t), ...]}`。"""
    groups = {}
    for episode in episodes:
        for match in episode.matches:
            for length in range(len(match.actions)):
                groups.setdefault(length, []).append((match, length))
    if not groups:
        raise ValueError("no demonstrations")
    return groups


def _batch_tensors(items):
    length = items[0][1]
    own = torch.stack([match.own_history for match, _ in items])
    opp = torch.stack([match.opp_history for match, _ in items])
    my = torch.tensor(
        [
            [action + 1 for action in match.actions[:length]]
            for match, _ in items
        ],
        dtype=torch.long,
    )
    opp_moves = torch.tensor(
        [
            [1 if move == rules.COOPERATE else 2 for move in match.opp_moves[:length]]
            for match, _ in items
        ],
        dtype=torch.long,
    )
    rs = torch.tensor([match.rs for match, _ in items], dtype=torch.long)
    labels = torch.tensor(
        [
            (
                match.labels[length]
                if len(match.labels) == len(match.actions)
                and match.labels[length] is not None
                else match.actions[length]
            )
            for match, _ in items
        ],
        dtype=torch.long,
    )
    return {
        "length": length,
        "own_history": own,
        "opp_history": opp,
        "my_decisions": my,
        "opp_decisions": opp_moves,
        "rs": rs,
        "labels": labels,
    }


def iter_batches(groups, batch_size=256, generator=None):
    """按长度遍历小批量；长度内部用 `generator` 打乱。"""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for length in sorted(groups):
        items = list(groups[length])
        if generator is None:
            order = list(range(len(items)))
        else:
            order = torch.randperm(len(items), generator=generator).tolist()
        for start in range(0, len(items), batch_size):
            chunk = [items[index] for index in order[start:start + batch_size]]
            yield _batch_tensors(chunk)


def train_behavior_cloning(
    policy,
    episodes,
    epochs=4,
    lr=1e-3,
    batch_size=256,
    max_grad_norm=1.0,
    seed=0,
    hard_example_weight=0.0,
    defect_weight=1.0,
    verbose=True,
):
    """用教师轨迹做交叉熵预训练，返回每轮指标。

    `hard_example_weight > 0` 时，对学生与教师意见不一致的样本加权，
    避免关键状态（如 AllD 下继续合作）被大量简单样本淹没。
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    groups = _group_demonstrations(episodes)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed * 7919 + epoch)
        policy.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch in iter_batches(groups, batch_size, generator):
            logits = policy(
                batch["own_history"],
                batch["opp_history"],
                batch["my_decisions"],
                batch["opp_decisions"],
                batch["rs"],
                batch["length"],
            )
            per_sample = F.cross_entropy(
                logits, batch["labels"], reduction="none"
            )
            if hard_example_weight > 0:
                mismatch = (
                    logits.argmax(dim=-1) != batch["labels"]
                ).to(per_sample.dtype)
                per_sample = per_sample * (
                    1.0 + hard_example_weight * mismatch
                )
            if defect_weight != 1.0:
                weights = torch.ones_like(batch["labels"], dtype=per_sample.dtype)
                weights[batch["labels"] == 1] = defect_weight
                per_sample = per_sample * weights
            loss = per_sample.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["labels"])
            total_correct += int(
                (logits.argmax(dim=-1) == batch["labels"]).sum().item()
            )
            total_count += len(batch["labels"])
        metrics = {
            "epoch": epoch,
            "loss": total_loss / max(total_count, 1),
            "accuracy": total_correct / max(total_count, 1),
            "samples": total_count,
        }
        history.append(metrics)
        if verbose:
            print(
                f"  bc epoch {epoch + 1}/{epochs}: "
                f"loss {metrics['loss']:.4f}, "
                f"accuracy {metrics['accuracy']:.3f} "
                f"({metrics['samples']} samples)"
            )
    return history


def bc_checkpoint_payload(policy, stage, metrics, demo_seasons):
    return {
        "stage": stage,
        "iteration": 0,
        "policy_config": policy.config(),
        "policy_state": policy.state_dict(),
        "predictor_state": None,
        "optimizer_state": None,
        "train_config": {
            "method": "behavior_cloning",
            "demo_seasons": demo_seasons,
        },
        "extra": {"bc": metrics},
    }


def _parse_seeds(text):
    return tuple(int(part) for part in text.split(",") if part.strip())


def build_parser():
    parser = argparse.ArgumentParser(description="行为克隆热启动（基线教师）")
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--seasons", type=int, default=20, help="教师轨迹赛季数")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--opponents-per-round", type=int, default=24)
    parser.add_argument("--eval-seeds", default="0,1")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--variants", type=int, default=None)
    parser.add_argument("--dagger-iterations", type=int, default=2)
    parser.add_argument("--student-seasons", type=int, default=10)
    parser.add_argument("--dagger-epochs", type=int, default=2)
    parser.add_argument("--dagger-lr", type=float, default=5e-4)
    parser.add_argument("--hard-example-weight", type=float, default=0.0)
    parser.add_argument("--defect-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-path", default=None, help="把纯 JSON 结果写到文件")
    return parser


def _evaluate(policy, opponents, config, stage, eval_seeds):
    from ipd.agents.baseline import baseline_agent
    from ipd.training.schedule import acceptance_report

    policy.eval()
    evaluation = evaluate_policy(
        policy,
        opponents,
        config,
        seeds=eval_seeds,
        baseline=baseline_agent(),
    )
    gate = acceptance_report(stage, evaluation)
    return evaluation, gate


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    from ipd.training.schedule import acceptance_report, stage_setup

    spec, opponents, config = stage_setup(
        args.stage, seed=args.seed, variants=args.variants
    )
    policy = IPDPolicy()
    teacher = BaselineTeacher()
    print(
        f"stage {args.stage}: collecting {args.seasons} teacher seasons "
        f"({len(opponents)} opponents, {args.opponents_per_round}/round)"
    )
    episodes = collect_demonstrations(
        opponents,
        config,
        seasons=args.seasons,
        seed=args.seed,
        opponents_per_round=args.opponents_per_round,
        teacher=teacher,
        verbose=True,
    )
    result = {
        "stage": args.stage,
        "demo_seasons": args.seasons,
        "policy_parameters": policy.num_parameters(),
        "evaluations": [],
    }
    best = {
        "score": None,
        "state": None,
        "metrics": None,
        "evaluation": None,
        "label": None,
    }

    def record(label, metrics):
        if args.no_eval:
            return None
        eval_seeds = _parse_seeds(args.eval_seeds)
        evaluation, gate = _evaluate(policy, opponents, config, args.stage, eval_seeds)
        result["evaluations"].append(
            {
                "label": label,
                "eval": evaluation,
                "gate": {"passed": gate.passed, "reasons": list(gate.reasons)},
            }
        )
        print(
            f"[{label}] eval {evaluation['policy_per_match']:.2f} "
            f"(baseline {evaluation['baseline_per_match']:.2f}), "
            f"gate {'PASS' if gate.passed else 'FAIL'}"
        )
        if gate.reasons:
            print("  gate reasons:", list(gate.reasons))
        if best["score"] is None or evaluation["policy_per_match"] > best["score"]:
            best["score"] = evaluation["policy_per_match"]
            best["state"] = {
                key: value.detach().clone()
                for key, value in policy.state_dict().items()
            }
            best["metrics"] = metrics
            best["evaluation"] = evaluation
            best["label"] = label
        return evaluation

    metrics = train_behavior_cloning(
        policy,
        episodes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        hard_example_weight=args.hard_example_weight,
        defect_weight=args.defect_weight,
        verbose=True,
    )
    result["bc"] = metrics
    evaluation = record("bc", metrics)

    for iteration in range(args.dagger_iterations):
        if evaluation is not None and (
            evaluation["policy_per_match"] >= evaluation["baseline_per_match"]
        ):
            print("BC already reaches baseline; skip DAgger.")
            break
        print(
            f"DAgger iteration {iteration + 1}/{args.dagger_iterations}: "
            f"collecting {args.student_seasons} student seasons"
        )
        student_episodes = collect_student_rollouts(
            policy,
            opponents,
            config,
            seasons=args.student_seasons,
            seed=args.seed + 101 + iteration,
            opponents_per_round=args.opponents_per_round,
            verbose=True,
        )
        labeled = label_episodes_with_teacher(student_episodes, teacher)
        print(f"  labeled {labeled} student states with the baseline teacher")
        episodes.extend(student_episodes)
        metrics = train_behavior_cloning(
            policy,
            episodes,
            epochs=args.dagger_epochs,
            lr=args.dagger_lr,
            batch_size=args.batch_size,
            seed=args.seed + 1000 + iteration,
            hard_example_weight=args.hard_example_weight,
            defect_weight=args.defect_weight,
            verbose=True,
        )
        result.setdefault("dagger", []).append(
            {
                "iteration": iteration,
                "student_seasons": args.student_seasons,
                "labeled": labeled,
                "bc": metrics,
            }
        )
        evaluation = record(f"dagger-{iteration + 1}", metrics)

    total_seasons = args.seasons + sum(
        item["student_seasons"] for item in result.get("dagger", [])
    )
    if best["state"] is not None:
        policy.load_state_dict(best["state"])
        print(
            f"restored best policy: {best['label']} "
            f"eval {best['score']:.2f}"
        )
    checkpoint = Path(args.checkpoint_dir) / f"stage{args.stage}_bc.pt"
    payload = bc_checkpoint_payload(
        policy, args.stage, best["metrics"] or metrics, total_seasons
    )
    payload["extra"] = {
        "bc": result.get("bc"),
        "dagger": result.get("dagger", []),
        "evaluations": result.get("evaluations", []),
    }
    save_checkpoint(checkpoint, payload)
    result["checkpoint"] = str(checkpoint)
    result["total_seasons"] = total_seasons
    if best["evaluation"] is not None:
        best_gate = acceptance_report(args.stage, best["evaluation"])
        best_gate_result = {
            "passed": best_gate.passed,
            "reasons": list(best_gate.reasons),
        }
    else:
        best_gate_result = None
    result["best"] = {
        "label": best["label"],
        "score": best["score"],
        "gate": best_gate_result,
    }
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"checkpoint: {checkpoint}")
    return result


if __name__ == "__main__":
    main()
