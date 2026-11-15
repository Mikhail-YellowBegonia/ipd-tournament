# 0006 分阶段引入对手复杂度，避免过早过拟合

- 日期：2026-11-14
- 状态：采纳

## 背景

自我对弈是训练的主渠道，但冷启动需要外部多样性对手。同时，Axelrod 库缺乏读历史能力，
不能定义我们最终要优化的环境。需要明确的导入顺序，使得强化学习在合理的难度阶梯上学习，
避免被 Axelrod 的多样性淹没。

## 决策

训练中对手复杂度分四个阶段导入。每阶段验证模型是否超过基线，再进入下一阶段。

### 阶段 1：基础对弈（第 1–2 周）

**对手层**：L0 + L1（不读历史）
- L0：AllC、AllD、Random(p)、交替、周期模式
- L1：TFT、TF2T、STFT、GTFT、Gradual、WSLS、Prober、Joss、ZD 系

**环境配置**：`TournamentConfig(hide_prob=0)`

**编码需求**：完整历史，处理补位 `'m'`（开局正常缺少历史的补位）

**目标**：学基础对弈逻辑（何时合作、何时背叛、什么是终局诱惑）

**验证条件**：
- 模型在此池子上相对基线无明显衰退
- 平均分、对各类对手的 regret 记录基线

### 阶段 2：固定规则读历史（第 3 周）

**新增**：L2（感知对手历史，但规则固定）
- 最后 k 手背叛检测、阶段切换反应、倒推背叛

**环境配置**：同阶段 1

**编码需求**：同阶段 1

**目标**：学到"对手可能基于简单规则读我的历史，需要预防"这个概念

**验证条件**：
- 相对基线的 regret 不增加（或保持）
- 对 L2 对手的表现开始优于对 L0/L1 的平均水位（说明学到了差异化）

### 阶段 3：智能读历史 + mask 隐藏（第 4–5 周）

**新增**：
- L3（读历史的对手，基于检测器库决策）
- 历史隐藏：`TournamentConfig(hide_prob=0.1, hide_scope="match")` 或 `"step"`

**编码需求**：
- 处理 mask（`'m'` 字符可能出现在任意位置，不仅仅是补位）
- n-gram 计数 + 嵌入表（架构已定，ADR 0002）
- 编码器对补位和随机隐藏无差别处理：都视为无信息

**目标**：学信誉成本；在不完整、不确定的信息下稳健决策

**关键关口**：
- 模型相对基线的优势是否还存在？
- 是否学到了防范读历史对手？
- 在隐藏历史的情况下性能衰退多少？

**如果此关口失败**：
- 回到阶段 2，增加训练数据量或调整学习率
- 或考虑跳到 ADR 0005 的纯进化策略路线

### 阶段 4：自我对弈 + 泛化检验（第 6–7 周）

**对手组成**：
- 主：种群自我对弈（3–5 个初始化各不相同的副本）
- 辅：L0–L3 仍保留，每轮混入 10–20%
- 验证：Axelrod 库的 10–15 个代表策略（EvolvedANN、EvolvedFSM、OmegaTFT 等），不计入训练信号，仅观测

**环境配置**：`TournamentConfig(hide_prob=0.1, hide_scope="match")`

**编码需求**：同阶段 3

**目标**：在自己的环境中找到局部最优；检验对 Axelrod 的泛化能力

**验证条件**：
- 种群平均分逐周上升
- 相对基线的优势稳定
- Axelrod 观测组的平均分不低于"基线 vs 同组对手"的平均

### 阶段 5：最终评估与微调（第 8 周）

**环境**：
- 留出测试池（队友独立编写，不看我方代码）
- Axelrod 库混合采样（20–30 个策略，模拟真实同学池的多样性）
- 多配比评估：L3 占比为 0%, 25%, 50%

**方法**：进化策略（ADR 0005）微调，适应度 = 多配比、5 种子的平均分

**验证条件**：
- 在留出池上超过基线
- 在各 L3 占比配置下都不崩溃

---

## 对应的代码结构

```python
# ipd/pool/registry.py（新增或改进）
def agents_for_stage(stage: int, seed=None) -> List[Agent]:
    """返回指定阶段的对手池。
    
    Args:
        stage: 1–4
        seed: 随机参数采样的种子
    
    Returns:
        Agent 列表
    """
    if stage == 1:
        return l0.agents() + l1.agents()
    elif stage == 2:
        return agents_for_stage(1) + l2.agents(seed=seed)
    elif stage == 3:
        return agents_for_stage(2) + l3.agents(seed=seed)
    elif stage == 4:
        return agents_for_stage(3)
    else:
        raise ValueError(f"Unknown stage: {stage}")

# ipd/training/schedule.py（新增）
TRAINING_SCHEDULE = {
    1: {
        "agents": agents_for_stage(1),
        "config": TournamentConfig(hide_prob=0.0),
        "weeks": 2,
        "checkpoint": "超过基线",
    },
    2: {
        "agents": agents_for_stage(2),
        "config": TournamentConfig(hide_prob=0.0),
        "weeks": 1,
        "checkpoint": "regret 不增",
    },
    3: {
        "agents": agents_for_stage(3),
        "config": TournamentConfig(hide_prob=0.1, hide_scope="match"),
        "weeks": 2,
        "checkpoint": "信誉成本已学，优势保留",
    },
    4: {
        "agents": agents_for_stage(4),
        "config": TournamentConfig(hide_prob=0.1, hide_scope="match"),
        "weeks": 2,
        "checkpoint": "自我对弈收敛，泛化可接受",
    },
}
```

## 理由与权衡

- **递进性**：早期只学稳定对手的基础特征，避免被多样性淹没。
- **信号清晰**：每阶段有明确的验收条件，卡壳时可诊断原因（编码不足、学习率、数据量）。
- **自我对弈有初值**：不是冷启动两个弱模型相互对弈，而是在 L0–L3 已训练过的基础上。
- **Axelrod 作验证，非主渠道**：保留泛化性检查，既不被其多样性主导，也不完全忽视。
- **读历史能力分离出来**：阶段 2（固定规则）和阶段 3（智能读历史）分开，诊断编码器是否真的学到了反应。

## 后续

- 立即：编写手工基线（issue #1），建立各池子下的基线得分表
- 立即：更新 `ipd/pool/registry.py`，支持阶段导入
- 编写 `ipd/training/schedule.py` 和训练主循环
- 队友并行：留出测试池（issue #11）

## 取代记录

（无）

