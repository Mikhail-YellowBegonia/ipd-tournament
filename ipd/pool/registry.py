"""对手池目录：标准版本与参数随机采样。"""

from dataclasses import dataclass
from typing import Callable

from ipd.pool import l0, l1, l2, l3
from ipd.sim.agent import Agent


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


def canonical():
    """每个族的标准版本，用于快速对照。"""
    tft = make_agent("tft")
    return [
        make_agent("all_c"),
        make_agent("all_d"),
        make_agent("rand", {"p": 0.5}),
        make_agent("periodic", {"pattern": "cb"}),
        tft,
        make_agent("stft"),
        make_agent("tf2t"),
        make_agent("gtft", {"g": 0.3}),
        make_agent("grim"),
        make_agent("wsls"),
        make_agent("joss", {"p": 0.1}),
        make_agent("gradual"),
        make_agent("prober"),
        make_agent("soft_majority"),
        make_agent("hard_majority"),
        make_agent("end_defect", {"k": 1}, [tft]),
        make_agent("end_defect", {"k": 3}, [tft]),
        make_agent("opening", {"pattern": "b"}, [make_agent("grim")]),
        make_agent("noisy", {"eps": 0.05}, [tft]),
        make_agent("reputation_gate", {"threshold": 0.25}, [tft]),
        make_agent("endgame_preempt", {"margin": 1, "min_rate": 0.5}, [tft]),
        make_agent("exploit_soft", {"threshold": 0.2}, [tft]),
        make_agent("opening_mirror", {"threshold": 0.5}, [tft]),
    ]


def _families(layers):
    return [f for f in FAMILIES if f.layer in layers]


def sample_agent(rng, layer):
    """随机采样一个指定层的 Agent，参数随机。"""
    fam = rng.choice(_families((layer,)))
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
