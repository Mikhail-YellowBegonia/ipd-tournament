"""留出测试池策略模板（issue #10，sub-issue 11.1）。

这是给**独立编写者**的示例模板：只依赖标准库，不导入 `ipd` 的任何模块。
复制本文件到 `ipd/pool/holdout/` 后，把 `STRATEGIES` 里的示例替换成自己的
13 个策略；也可以保留示例作为对照。

接口（九个参数，参数名必须一致）::

    def strategy(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
        ...

- `rs`：当前轮次 0–7；`length`：本场已完成回合数 0–24；
- `my_decisions`/`opp_decisions`：本场双方此前动作，元素 `"c"`/`"b"`；
- `repay`：25×4 的 `(T,S,R,P)` 元组，`repay[length]` 是当前回合的收益行；
- `opp_history_decisions`：对手最近 20 场（20×25）动作，`"m"` 表示补位/隐藏；
- `opp_opp_history_decisions`：上述各场中对手的对手（也就是我方）的动作。

返回值只能是 `"c"` 或 `"b"`。整个函数体包在 `try/except` 内，异常返回 `"c"`。
不得保存跨调用状态（不写 `global`/`nonlocal`），不得读写文件/联网，不得
调用 `random.seed()`。

标签（`tags`）从以下集合选取，必须如实反映行为：

- `basic`：固定规则/经典策略；
- `cooperative`：以合作为主；
- `hostile`：以背叛为主；
- `conditional`：根据本场对手此前动作/合作率/比分改变行为；
- `history`：实际读取并使用了 `opp_history_decisions` 或 `opp_opp_history_decisions`；
- `endgame`：会在终局（最后 1–5 手）切换为背叛；
- `noisy`：有随机性；
- `mixed`：在多个基础策略/模式间切换。

完整编写与审查流程见 `docs/eval/holdout_pool_guidelines.md`。
"""

import random

STRATEGIES = []


def register(name, tags):
    """把策略登记进 `STRATEGIES`；名称全局唯一并以 `ho_` 开头。"""
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


def _cooperation_rate(seq):
    """序列里的合作率；跳过 `m`，空序列返回 0.5。"""
    valid = [move for move in seq if move != "m"]
    if not valid:
        return 0.5
    return valid.count("c") / len(valid)


@register("ho_template_conditional", ("conditional",))
def conditional_threshold(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                          repay, opp_history_decisions, opp_opp_history_decisions):
    """对手在本场的合作率低于 50% 就背叛，否则合作。"""
    try:
        if length == 0:
            return "c"
        return "b" if _cooperation_rate(opp_decisions) < 0.5 else "c"
    except Exception:
        return "c"


@register("ho_template_history", ("history", "conditional"))
def history_type_reader(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions, opp_opp_history_decisions):
    """用对手最近 20 场的合作率判断类型：合作者就合作，敌对者就背叛。"""
    try:
        rate = _cooperation_rate(
            move for row in opp_history_decisions for move in row
        )
        return "c" if rate >= 0.5 else "b"
    except Exception:
        return "c"


@register("ho_template_endgame", ("basic", "endgame"))
def endgame_defect(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """前 24 手合作，最后一手背叛（最朴素的终局策略）。"""
    try:
        return "b" if length >= 24 else "c"
    except Exception:
        return "c"


@register("ho_template_noisy_tft", ("conditional", "noisy"))
def noisy_tft(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """带 5% 翻转概率的以牙还牙。"""
    try:
        if length == 0:
            return "c"
        move = opp_decisions[-1]
        if random.random() < 0.05:
            return "b" if move == "c" else "c"
        return move
    except Exception:
        return "c"
