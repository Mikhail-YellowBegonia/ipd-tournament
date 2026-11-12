"""L1：只看本场动作的反应式策略。"""

import random

C, D = "c", "b"


def tft():
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return opp[-1] if length else C
    return s


def stft():
    """首手背叛的 TFT。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return opp[-1] if length else D
    return s


def tf2t():
    """对手连续两次背叛才报复。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return D if length >= 2 and opp[-1] == D and opp[-2] == D else C
    return s


def gtft(g=0.3):
    """宽容 TFT：对手背叛后仍以概率 g 合作。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        if not length or opp[-1] == C:
            return C
        return C if random.random() < g else D
    return s


def grim():
    """对手背叛一次后永久背叛。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return D if D in opp else C
    return s


def wsls():
    """赢了保持、输了换（Pavlov）：上一手双方相同则合作。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        if not length:
            return C
        return C if my[-1] == opp[-1] else D
    return s


def memory_one(p_cc, p_cd, p_dc, p_dd, p0=1.0):
    """通用记忆一步策略：按上一手 (我, 对手) 决定合作概率。

    覆盖 TFT、WSLS、GTFT、零行列式（ZD）等一大类策略。
    """
    probs = {(C, C): p_cc, (C, D): p_cd, (D, C): p_dc, (D, D): p_dd}

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        p = probs[(my[-1], opp[-1])] if length else p0
        return C if random.random() < p else D
    return s


def joss(p=0.1):
    """TFT，但在对手合作后以概率 p 偷偷背叛。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        if length and opp[-1] == D:
            return D
        return D if random.random() < p else C
    return s


def gradual():
    """对手第 n 次背叛后连续背叛 n 次，再合作两次以示和解。

    无跨调用状态，每次调用从本场开头重放自己的状态机。
    """
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        punish = calm = n_def = 0
        for t in range(length + 1):
            if punish:
                move = D
                punish -= 1
                if not punish:
                    calm = 2
            elif calm:
                move = C
                calm -= 1
            else:
                move = C
            if t == length:
                return move
            if opp[t] == D:
                n_def += 1
                if not punish and not calm:
                    punish = n_def
    return s


def prober():
    """先出 D、C、C；若对手第 2、3 手都合作，判定可剥削并永久背叛，否则 TFT。"""
    opening = (D, C, C)

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        if length < 3:
            return opening[length]
        if opp[1] == C and opp[2] == C:
            return D
        return opp[-1]
    return s


def soft_majority():
    """对手合作次数不少于背叛次数时合作。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return C if opp.count(C) >= opp.count(D) else D
    return s


def hard_majority():
    """对手背叛次数不少于合作次数时背叛（首手背叛）。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return D if opp.count(D) >= opp.count(C) else C
    return s
