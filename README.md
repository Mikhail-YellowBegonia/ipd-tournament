# 有限重复囚徒困境策略（计算概论 B 大作业）

用强化学习训练一个策略，在 25 回合、收益分阶段变化、可读取对手历史的重复囚徒困境锦标赛中取得高总分。

- 截止：12 月 15 日 15:10（教学网提交）
- 规则原文摘要：[docs/reference/course_rules.md](docs/reference/course_rules.md)

## 硬约束（提交物）

- 只用标准库（具体白名单以调试包 README 为准，**待确认**）
- 单次决策 ≤ 0.5 秒；不得跨调用保存可变状态（含缓存）
- 异常/超时/非法返回按合作处理，同一种子累计 3 次后永久合作
- 训练阶段不受限；**评测阶段必须完全对齐考试环境**

## 目录

```
docs/
  reference/     外部资料原文（课程规则、调试包 README 等），只读
  design/        设计文档：问题分析、系统架构
  decisions/     决策记录（ADR），一个决策一个文件，只追加不改写
  experiments/   实验日志，一个实验一个文件
  roadmap.md     任务清单、阻塞项、待确认问题
debug_kit/       学生调试包原样存放（收到后）
ipd/             代码包：sim（模拟器）、pool（对手池）……
tests/           单元测试
```

## 常用命令

```bash
python -m unittest discover -s tests -t .   # 运行测试
python -m ipd.sim.run                        # 标准对手跑 5 个种子的联赛
python -m ipd.sim.run --sample 40 --top 15   # 加 40 个随机参数对手
```

完整代码目录规划见 [docs/design/architecture.md](docs/design/architecture.md#代码目录规划)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/roadmap.md](docs/roadmap.md) | 现在能做什么、什么被阻塞 |
| [docs/design/problem_analysis.md](docs/design/problem_analysis.md) | 收益结构、信誉机制、计算约束 |
| [docs/design/architecture.md](docs/design/architecture.md) | 教师-学生架构、历史编码器、对手池、信誉塑形 |
| [docs/decisions/](docs/decisions/) | 已做出的关键决策及理由 |
| [docs/experiments/](docs/experiments/) | 实验记录 |
