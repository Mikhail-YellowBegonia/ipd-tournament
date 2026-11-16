"""对手池目录：标准版本、参数随机采样与分阶段训练配置。"""

import random
from dataclasses import dataclass
from typing import Callable

from ipd.pool import l0, l1, l2, l3
from ipd.sim.agent import Agent
from ipd.sim.config import TournamentConfig


@dataclass(frozen=True)
class Family:
    name: str
    layer: str
    factory: Callable
    # rng -> 参数字典；None 表示无参数
    sample: Callable = None
    # 包装器族需要基础策略
    wrapper: bool = False


def _r(x):
    return round(x, 2)


FAMILIES = [
    Family("all_c", "L0", l0.all_c),
    Family("all_d", "L0", l0.all_d),
    Family("rand", "L0", l0.rand, lambda rng: {"p": _r(rng.uniform(0.1, 0.9))}),
    Family("periodic", "L0", l0.periodic,
           lambda rng: {"pattern": rng.choice(["cb", "bc", "ccb", "cbb", "cccb"])}),
    Family("tft", "L1", l1.tft),
    Family("stft", "L1", l1.stft),
    Family("tf2t", "L1", l1.tf2t),
    Family("gtft", "L1", l1.gtft, lambda rng: {"g": _r(rng.uniform(0.05, 0.5))}),
    Family("grim", "L1", l1.grim),
    Family("wsls", "L1", l1.wsls),
    Family("memory_one", "L1", l1.memory_one,
           lambda rng: {k: _r(rng.random()) for k in ("p_cc", "p_cd", "p_dc", "p_dd", "p0")}),
    Family("joss", "L1", l1.joss, lambda rng: {"p": _r(rng.uniform(0.02, 0.2))}),
    Family("gradual", "L1", l1.gradual),
    Family("prober", "L1", l1.prober),
    Family("soft_majority", "L1", l1.soft_majority),
    Family("hard_majority", "L1", l1.hard_majority),
    Family("end_defect", "L2", l2.end_defect, lambda rng: {"k": rng.randint(1, 5)}, wrapper=True),
    Family("opening", "L2", l2.opening,
           lambda rng: {"pattern": rng.choice(["b", "bb", "bc", "cb"])}, wrapper=True),
    Family("noisy", "L2", l2.noisy, lambda rng: {"eps": _r(rng.uniform(0.01, 0.15))}, wrapper=True),
    Family("faulty", "L2", l2.faulty, lambda rng: {"p": _r(rng.uniform(0.005, 0.05))}, wrapper=True),
    # by_phase 需要三个基础策略，单独处理
    Family("by_phase", "L2", l2.by_phase, wrapper=True),
    Family("reputation_gate", "L3", l3.reputation_gate,
           lambda rng: {"threshold": _r(rng.uniform(0.1, 0.4))}, wrapper=True),
    Family("endgame_preempt", "L3", l3.endgame_preempt,
           lambda rng: {"margin": rng.randint(0, 2), "min_rate": _r(rng.uniform(0.3, 0.7))}, wrapper=True),
    Family("exploit_soft", "L3", l3.exploit_soft,
           lambda rng: {"threshold": _r(rng.uniform(0.1, 0.4))}, wrapper=True),
    Family("opening_mirror", "L3", l3.opening_mirror,
           lambda rng: {"threshold": _r(rng.uniform(0.3, 0.7))}, wrapper=True),
]

FAMILY_BY_NAME = {f.name: f for f in FAMILIES}


def _label(name, params, base_labels=()):
    label = name
    if params:
        label += "(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"
    if base_labels:
        label += "[" + "|".join(base_labels) + "]"
    return label


def make_agent(name, params=None, bases=()):
    """按族名、参数和基础策略（Agent 列表）构造 Agent。"""
    fam = FAMILY_BY_NAME[name]
    params = dict(params or {})
    base_fns = [b.fn for b in bases]
    if fam.name == "by_phase":
        fn = fam.factory(base_fns)
    elif fam.wrapper:
        (base_fn,) = base_fns
        fn = fam.factory(base_fn, **params)
    else:
        fn = fam.factory(**params)
    label = _label(name, params, [b.name for b in bases])
    record = dict(params)
    if bases:
        record["bases"] = [b.name for b in bases]
    return Agent(name=label, fn=fn, family=name, layer=fam.layer, params=record)


# 每个族"标准版本"的固定参数；未列出的族无参数。与 Family.sample 的默认取值一致。
STANDARD_PARAMS = {
    "rand": {"p": 0.5},
    "periodic": {"pattern": "cb"},
    "gtft": {"g": 0.3},
    # 与 TFT 等价：上一手 (c,c) 继续合作，(d,d) 继续背叛，(d,c) 回到合作
    "memory_one": {"p_cc": 1, "p_cd": 0, "p_dc": 1, "p_dd": 0, "p0": 1},
    "joss": {"p": 0.1},
    "end_defect": {"k": 1},
    "opening": {"pattern": "b"},
    "noisy": {"eps": 0.05},
    "faulty": {"p": 0.02},
    "reputation_gate": {"threshold": 0.25},
    "endgame_preempt": {"margin": 1, "min_rate": 0.5},
    "exploit_soft": {"threshold": 0.2},
    "opening_mirror": {"threshold": 0.5},
}

# 包装器族标准版本所包的基础策略（族名）。
STANDARD_BASES = {
    "end_defect": ("tft",),
    "opening": ("grim",),
    "noisy": ("tft",),
    "faulty": ("tft",),
    "by_phase": ("tft", "tf2t", "grim"),
    "reputation_gate": ("tft",),
    "endgame_preempt": ("tft",),
    "exploit_soft": ("tft",),
    "opening_mirror": ("tft",),
}


def standard_agent(name):
    """构造某一族的标准版本（固定参数与基础策略）。"""
    bases = [make_agent(base) for base in STANDARD_BASES.get(name, ())]
    return make_agent(name, STANDARD_PARAMS.get(name, {}), bases)


def canonical():
    """每个族的标准版本，用于快速对照。"""
    tft = make_agent("tft")
    return [
        standard_agent("all_c"),
        standard_agent("all_d"),
        standard_agent("rand"),
        standard_agent("periodic"),
        tft,
        standard_agent("stft"),
        standard_agent("tf2t"),
        standard_agent("gtft"),
        standard_agent("grim"),
        standard_agent("wsls"),
        standard_agent("joss"),
        standard_agent("gradual"),
        standard_agent("prober"),
        standard_agent("soft_majority"),
        standard_agent("hard_majority"),
        standard_agent("end_defect"),
        make_agent("end_defect", {"k": 3}, [tft]),
        standard_agent("opening"),
        standard_agent("noisy"),
        standard_agent("reputation_gate"),
        standard_agent("endgame_preempt"),
        standard_agent("exploit_soft"),
        standard_agent("opening_mirror"),
    ]


def _families(layers):
    return [f for f in FAMILIES if f.layer in layers]


def sample_family(rng, fam):
    """按某个族随机采样一个 Agent：参数与基础策略都随机。"""
    params = fam.sample(rng) if fam.sample else {}
    if fam.name == "by_phase":
        bases = [sample_agent(rng, "L1") for _ in range(3)]
    elif fam.layer == "L3":
        # 读历史的策略包在反应式或回合感知策略外，偶尔叠加另一个读历史策略
        bases = [sample_agent(rng, rng.choice(["L1", "L1", "L2", "L3"]))]
    elif fam.wrapper:
        # 包装器可以叠加：基础策略有一定概率本身也是 L2
        bases = [sample_agent(rng, "L2" if rng.random() < 0.2 else rng.choice(["L0", "L1", "L1"]))]
    else:
        bases = []
    return make_agent(fam.name, params, bases)


def sample_agent(rng, layer):
    """随机采样一个指定层的 Agent，参数随机。"""
    return sample_family(rng, rng.choice(_families((layer,))))


DEFAULT_MIX = {"L0": 0.10, "L1": 0.35, "L2": 0.30, "L3": 0.25}


def uniquify(agents):
    """重名的 Agent 依次加后缀 #2、#3……，保证同场名字唯一。"""
    out, used = [], set()
    for agent in agents:
        stem = agent.name
        head, sep, tail = stem.rpartition("#")
        if sep and tail.isdigit():
            stem = head
        name, k = agent.name, 1
        while name in used:
            k += 1
            name = f"{stem}#{k}"
        used.add(name)
        if name != agent.name:
            agent = Agent(name, agent.fn, agent.family, agent.layer, agent.params)
        out.append(agent)
    return out


def sample_population(rng, n, mix=None):
    """按层配比采样 n 个 Agent，名字唯一。"""
    mix = mix or DEFAULT_MIX
    layers, weights = zip(*mix.items())
    return uniquify([sample_agent(rng, rng.choices(layers, weights)[0]) for _ in range(n)])


# ADR 0006 的四个训练阶段各含哪些对手层。
STAGE_LAYERS = {
    1: ("L0", "L1"),
    2: ("L0", "L1", "L2"),
    3: ("L0", "L1", "L2", "L3"),
    4: ("L0", "L1", "L2", "L3"),
}

# 阶段 3–4 引入历史隐藏（概率、粒度见 TournamentConfig）。
STAGE_HIDE_PROB = {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.1}

# 阶段池中每个可随机族在给定 seed 时额外生成的变体数。
STAGE_VARIANTS = 2


def _randomizable(fam):
    """该族是否有随机参数或随机基础策略。

    无参数、无包装的族（如 all_c、tft）加变体只会重复标准版本，因此跳过。
    """
    return fam.sample is not None or fam.name == "by_phase"


def agents_for_stage(stage, seed=None, variants=None):
    """按 ADR 0006 返回指定训练阶段的对手池。

    每个阶段包含其允许层内**所有族**的标准版本；给定 seed 时，再为每个
    有随机参数的族生成 `variants`（默认 2）个随机参数变体。

    Args:
        stage: 训练阶段编号 1–4。阶段 4（自我对弈为主）沿用阶段 3 的对手。
        seed: 随机参数采样的种子；None 表示只返回标准版本。
        variants: 每个可随机族生成的变体数；None 用 STAGE_VARIANTS，0 表示不加。

    Returns:
        Agent 列表，名字唯一。规模（variants=2）：
        阶段 1 = 16 / 26，阶段 2 = 21 / 41，阶段 3、4 = 25 / 53
        （斜杠前为 seed=None，斜杠后为给定 seed）。

    Raises:
        ValueError: stage 不在 1–4。
    """
    if stage not in STAGE_LAYERS:
        raise ValueError(f"Unknown stage: {stage}")
    variants = STAGE_VARIANTS if variants is None else variants
    layers = STAGE_LAYERS[stage]
    families = _families(layers)
    agents = [standard_agent(fam.name) for fam in families]
    if seed is not None and variants > 0:
        rng = random.Random(seed)
        for fam in families:
            if _randomizable(fam):
                agents.extend(sample_family(rng, fam) for _ in range(variants))
    return uniquify(agents)


def config_for_stage(stage):
    """按 ADR 0006 返回指定训练阶段的 TournamentConfig。

    阶段 1–2 使用完整历史；阶段 3–4 引入整场历史隐藏（hide_prob=0.1）。
    其余字段保持 TournamentConfig 默认值（待调试包对齐，见 docs/roadmap.md）。

    Raises:
        ValueError: stage 不在 1–4。
    """
    if stage not in STAGE_LAYERS:
        raise ValueError(f"Unknown stage: {stage}")
    if stage <= 2:
        return TournamentConfig(hide_prob=STAGE_HIDE_PROB[stage])
    return TournamentConfig(hide_prob=STAGE_HIDE_PROB[stage], hide_scope="match")
