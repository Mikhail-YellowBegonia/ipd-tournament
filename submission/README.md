# 基线兜底提交包

## 这是什么

`submission/strategy.py` 是 `ipd/agents/baseline.py` 的**独立打包版本**：

- 只依赖标准库（实际上没有任何 import）；
- 保留考试九参数 `strategy` 函数，返回 `"c"`/`"b"`；
- 无跨调用可变状态，不读写文件/联网，不修改传入数据；
- 单次决策约 **0.074 ms**（完整 20×25 历史、300 次平均），远低于 0.5 s；
- 在阶段 1 池的 2 轮锦标赛中与 `ipd/agents/baseline.py` 的分数完全一致，0 故障。

它是**兜底方案**，不是项目最终目标。最终若提交学习模型，应替换本文件，而不是
把基线和学习模型拼在一起。

## 本地验证

调试包到手后：

```bash
# 1. 把本文件复制到调试包目录，保留 strategy.py 的名字
cp submission/strategy.py /path/to/debug_kit/strategy.py
cd /path/to/debug_kit

# 2. 按课程要求检查
python validate_submission.py strategy.py
python run_local.py
python run_local.py --league
```

没有调试包时，仓库里可以这样验证：

```bash
.venv/bin/python -m unittest tests.test_baseline_standalone
```

## 正式提交

按课程要求改成 `gXX_team_name.py` / `gXX_team_name.txt`（`XX` 为两位组号，
`team_name` 只含英文字母、数字、下划线），函数名仍为 `strategy`：

```bash
cp submission/strategy.py g07_our_team.py
cp submission/description_template.txt g07_our_team.txt
```

说明文档草稿在 `submission/description_template.txt`，请补上成员与分工后压到
300 字以内。

## 参考成绩

- 阶段 1 池（26 对手 + 基线，8 轮，5 种子）：约 **47.8 分/场**；
- 留出池（30 抽样/种子，5 种子）：**47.94 分/场**（实验 0007）。

如果学习模型最终没有稳定超过这个水位，提交本包比提交一个更弱的模型更稳。
