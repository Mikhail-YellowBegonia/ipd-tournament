# 决策记录（ADR）

一个决策一个文件：`NNNN-short-title.md`。已记录的决策不改写；推翻时新建一条并标注"取代 NNNN"。

模板：

```
# NNNN 标题
- 日期：YYYY-MM-DD
- 状态：采纳 / 取代 NNNN / 已被 NNNN 取代

## 背景
## 决策
## 理由与权衡
## 后果
```

## 索引

- [0001 以强化学习为主路线，手工基线仅作参照尺](0001-rl-main-route.md)
- [0002 分层历史编码：先线性计数，再非线性](0002-hierarchical-history-encoder.md)
- [0003 教师-学生蒸馏；评测完全对齐考试环境](0003-distillation-and-eval-alignment.md)
- [0004 信誉成本以闭环为主、塑形项为对冲](0004-reputation-shaping.md)
