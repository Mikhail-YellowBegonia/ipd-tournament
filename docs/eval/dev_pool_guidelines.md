# Dev 测试池编写指南（T3）

- 日期：2026-11-17
- 对应计划：[plan 0001](../plans/0001-tail-first.md) T3
- 关联决策：ADR 0009（尾部优先、规则先于模型）
- 实现：`ipd/pool/dev/`（60 个策略）、`ipd/eval/diagnose.py`（`--pool dev`）、
  `tests/test_dev_pool.py`
- 生成契约（本地）：`.dev_staging/CONTRACT.md`、
  `.dev_staging/check_dev_module.py`

## 目的

130 个留出策略（实验 0007）已经用来发现 `guard_open`；继续在它上面调参
会把测试集变成验证集。Dev 池是**开发/验证集**：

- 60 个独立编写、未看过仓库实现的策略；
- 规则/分类器的迭代在 dev 池 + 标准池上做；
- **130 留出池冻结**，只在最终 go/no-go 时跑；
- 与留出池一样不进入训练（`agents_for_stage()` 不含它们）。

## 接口与硬性要求

与留出池完全一致：九参数函数、只允许标准库（`random`/`math`/`collections`/
`itertools`/`functools`）、无 `global`/`nonlocal`、无跨调用状态、不改输入、
不读写文件/联网/打印、不调用 `random.seed()`、异常返回 `"c"`。

收益表与历史语义见 `docs/reference/course_rules.md` 与
`docs/eval/holdout_pool_guidelines.md`。

## 规模与覆盖

- 6 个模块 × 10 个策略 = **60**；名称以 `dv_` 开头。
- 标签下限（按 60 个计）：
  - `history` >= 12（20%）
  - `conditional` >= 18（30%）
  - `endgame` >= 6（10%）
  - `noisy` >= 6（10%）
- 主题分配：
  1. `dev_basic_a`：经典/极端合作/敌对；
  2. `dev_basic_b`：周期/随机/概率/阈值；
  3. `dev_conditional_a`：本场条件 + 终局；
  4. `dev_history_a`：读历史/类型/声誉；
  5. `dev_mixed_a`：多模式 + 噪声；
  6. `dev_mixed_b`：混合 + 历史 + 终局 + 噪声。

## 生成与审查

1. 6 个隔离 subagent，各自只读 `.dev_staging/CONTRACT.md` 与自己的主题；
   不读 `ipd/`、`tests/`、`docs/`、`.holdout_staging/`、其他作者文件。
2. 每个作者写 10 个策略，运行 `check_dev_module.py` 到 0 error。
3. 仓库内复查：
   - `tests/test_dev_pool.py` 的 AST/签名/标签/边界；
   - 行为探针：history 真的读历史、conditional 响应本场、endgame 末手背叛、
     noisy 重复调用出现两种动作；
   - 指纹多样性 >= 54/60；
   - 全池 + 标准池 0 故障；
   - 与 `agents_for_stage()` 无交集。

## 用法

```bash
# dev 池 5 seeds（规则/模型迭代）
python3 -m ipd.eval.diagnose --agent <候选> --pool dev

# dev 池单种子快速筛查
python3 -m ipd.eval.diagnose --agent <候选> --pool dev --seeds 0

# 测试
python3 -m unittest tests.test_dev_pool
```

注意：dev 池上的提升**不等于**最终提升；最终决策仍以 130 留出池的
全池 5 seeds + 克隆对照为准（计划 0001 的 T1 门槛）。
