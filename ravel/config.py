"""Configuration dataclasses for RAVEL V2.1 and a YAML loader with dotted overrides."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

import yaml


@dataclass
class AudioCfg:
    sample_rate: int = 24000
    tick_samples: int = 480
    ticks_per_slot: int = 2
    n_fft: int = 1024
    n_mels: int = 80
    f_min: float = 40.0
    f_max: float = 12000.0

    @property
    def slot_samples(self) -> int:
        return self.tick_samples * self.ticks_per_slot


@dataclass
class CodecCfg:
    name: str = "encodec_24khz"
    hop: int = 320
    frames_per_slot: int = 3
    n_q: int = 4
    vocab: int = 1024
    bandwidth: float = 3.0


@dataclass
class APFCfg:
    d_model: int = 384
    kernel: int = 3
    dilations: list = field(default_factory=lambda: [1, 2, 4, 8, 1, 2, 4, 8])
    dropout: float = 0.0
    apc_offset_ticks: int = 3


@dataclass
class CoreCfg:
    d_h: int = 640
    d_a: int = 640
    d_i: int = 320
    d_m: int = 512
    d_couple: int = 192
    d_act: int = 64
    d_intent: int = 0        # optional AI-private intent input (used by the simulator: pending content)
    slow_every: int = 10
    ffn_mult: int = 3
    lru_c: float = 8.0


@dataclass
class FeedbackCfg:
    d_f: int = 320


@dataclass
class RIFCfg:
    enabled: bool = True
    listener_heads: bool = True
    d_hidden: int = 640
    d_head: int = 384
    n_layers: int = 2
    horizon_slots: int = 12
    n_ensemble: int = 4
    bootstrap: float = 0.8
    gamma: float = 0.96
    events: list = field(default_factory=lambda: ["bargein", "turn_abandon", "reprompt"])


@dataclass
class PolicyCfg:
    w_overlap: float = 1.0
    beta_uncert: float = 0.5
    alpha_relevance: float = 0.2
    kappa_compute: float = 0.01
    temperature: float = 0.1
    prior_weight: float = 0.0


@dataclass
class PlannerCfg:
    enabled: bool = True
    n_slots: int = 8
    d_slot: int = 512
    n_layers: int = 2
    n_heads: int = 8
    probe_codebooks: int = 1


@dataclass
class DNAGCfg:
    d_model: int = 640


@dataclass
class TrainCfg:
    checkpoint_activations: bool = False


@dataclass
class RavelConfig:
    name: str = "ravel_tiny_v2.1"
    audio: AudioCfg = field(default_factory=AudioCfg)
    codec: CodecCfg = field(default_factory=CodecCfg)
    actions: list = field(
        default_factory=lambda: ["silent", "yield", "backchannel", "speak", "speak_slow", "speak_fast", "speak_emph"]
    )
    apf: APFCfg = field(default_factory=APFCfg)
    core: CoreCfg = field(default_factory=CoreCfg)
    feedback: FeedbackCfg = field(default_factory=FeedbackCfg)
    rif: RIFCfg = field(default_factory=RIFCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    planner: PlannerCfg = field(default_factory=PlannerCfg)
    dnag: DNAGCfg = field(default_factory=DNAGCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def to_dict(self) -> dict:
        return asdict(self)


def _fill(dc_type, data: dict):
    kwargs = {}
    valid = {f.name: f for f in fields(dc_type)}
    for k, v in (data or {}).items():
        if k not in valid:
            raise KeyError(f"Unknown config key '{k}' for {dc_type.__name__}")
        sub = _SUBCONFIGS.get(k) if dc_type is RavelConfig else None
        kwargs[k] = _fill(sub, v) if sub is not None else v
    return dc_type(**kwargs)


_SUBCONFIGS = {
    "audio": AudioCfg, "codec": CodecCfg, "apf": APFCfg, "core": CoreCfg, "feedback": FeedbackCfg,
    "rif": RIFCfg, "policy": PolicyCfg, "planner": PlannerCfg, "dnag": DNAGCfg, "train": TrainCfg,
}


def from_dict(data: dict) -> RavelConfig:
    return _fill(RavelConfig, data)


def apply_overrides(cfg: RavelConfig, overrides: list[str] | dict | None) -> RavelConfig:
    """Apply overrides like ["core.d_h=256", "rif.enabled=false"] (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    if not overrides:
        return cfg
    items = overrides.items() if isinstance(overrides, dict) else (o.split("=", 1) for o in overrides)
    for key, val in items:
        if isinstance(val, str):
            val = yaml.safe_load(val)
        obj: Any = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"Unknown config key '{key}'")
        setattr(obj, parts[-1], val)
    return cfg


def load_config(path: str, overrides: list[str] | dict | None = None) -> RavelConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return apply_overrides(from_dict(data), overrides)


def micro_overrides() -> dict:
    """A ~3M-parameter configuration used for CPU-budget ablations (same architecture, smaller widths)."""
    return {
        "name": "ravel_micro_v2.1",
        "apf.d_model": 160, "apf.dilations": [1, 2, 4, 8],
        "core.d_h": 192, "core.d_a": 192, "core.d_i": 128, "core.d_m": 128, "core.d_couple": 64,
        "core.d_act": 32, "core.ffn_mult": 2,
        "feedback.d_f": 96,
        "rif.d_hidden": 192, "rif.d_head": 128,
        "planner.d_slot": 192, "planner.n_layers": 1, "planner.n_heads": 4,
        "dnag.d_model": 256,
    }


def config_summary(cfg: RavelConfig) -> str:
    def walk(o, prefix=""):
        out = []
        for f in fields(o):
            v = getattr(o, f.name)
            if is_dataclass(v):
                out += walk(v, prefix + f.name + ".")
            else:
                out.append(f"{prefix}{f.name}={v}")
        return out
    return "\n".join(walk(cfg))
