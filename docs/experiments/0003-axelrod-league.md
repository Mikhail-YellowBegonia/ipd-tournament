# Axelrod 策略加入联赛
- 关联决策：ADR 0005

## 假设
axelrod 库中演化出来的策略（EvolvedFSM、EvolvedANN 等）在本题收益表下仍然强，可作为"复杂死算法"对手。

## 设置
- 对手：23 个标准对手 + 224 个 axelrod 策略（排除作弊与 long_run_time），共 247 个
- 种子 0（单种子，约 210 秒），`TournamentConfig(timeout=None)`，其余默认
- 命令：`.venv/bin/python -W ignore -m ipd.sim.run --axl --seeds 0`（实验时 timeout=None，`run` 默认 0.5 秒，结果无差别：最慢策略均值约 1 ms）

## 结果（每场平均分）

| 名次 | 策略 | 平均分 |
|---:|---|---:|
| 1 | axl:EvolvedFSM16Noise05 | 50.60 |
| 2 | axl:EvolvedFSM6 | 50.03 |
| 3 | axl:EvolvedANN5 | 49.56 |
| 4 | axl:OmegaTFT | 49.14 |
| 7 | axl:EvolvedANN | 48.20 |
| 10 | reputation_gate[tft]（我方 L3） | 46.85 |
| 18 | gradual | 46.00 |
| 81 | tft | 41.10 |
| 216 | all_d | −5.68 |

全程 0 故障。

## 结论与下一步
- 演化策略即使在不匹配的收益表下训练，仍占据前列；证实它们是强对手，也说明小模型足够。
- 池子中 axelrod 策略占九成以上，评估时必须先做行为指纹去重（issue #2），否则结果由 axelrod 主导。
- 单种子，未做多配比；仅作为第一印象。
