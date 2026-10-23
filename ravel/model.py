"""RAVEL V2.1 — Reciprocal Acoustic Variational Exchange Lattice (integrated model).

Per 20 ms tick t (slot s = t // 2):
    mic_mel_t, ref_mel_t  = CausalLogMel(mic), CausalLogMel(AI output reference)
    z_t                   = APF(mic_mel_{<=t}, ref_mel_{<=t})
    f_t                   = SMF(tokens emitted in slot s, a_s, phase)
    (H, I, A, M)_t        = Core((H, I, A, M)_{t-1}, z_t, p_user_t, f_t, e(a_s), mean(P))
At the end of slot s (decision d = s, chooses a_{s+1} and commits audio for slot s+1):
    R(a)                  = RIF(H, A, I, M, a)   for every candidate a
    a_{s+1}               = Policy(R, prior)      (or the logged action under teacher forcing)
    P                     = Planner(Shift(P), A, I, e(a_{s+1}), stopgrad(RIF summary of a_{s+1}))
    tokens_{s+1}          = DNAG(P[0], last emitted frame)  -> frozen codec decoder -> 960 samples
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .acoustic_encoder import AcousticPerceptionField
from .acoustic_planner import DirectSlotHead, ShortHorizonPlanner, plan_probe_logits
from .actions import ActionEmbedding, ActionSpace
from .audio_generator import DirectNeuralAudioGenerator
from .config import RavelConfig
from .conversational_core import DualStateCore
from .features import CausalLogMel
from .feedback import SelfMonitoringFeedback
from .influence_field import ReciprocalInfluenceField
from .policy import InteractionPolicy


class RAVEL(nn.Module):
    def __init__(self, cfg: RavelConfig):
        super().__init__()
        self.cfg = cfg
        a, c, co = cfg.audio, cfg.codec, cfg.core
        self.space = ActionSpace(cfg.actions)
        C = len(self.space)
        self.tps = a.ticks_per_slot
        assert a.slot_samples == c.hop * c.frames_per_slot, "slot must map to an integer number of codec frames"
        self.mel = CausalLogMel(a.sample_rate, a.tick_samples, a.n_fft, a.n_mels, a.f_min, a.f_max)
        self.apf = AcousticPerceptionField(a.n_mels, cfg.apf.d_model, cfg.apf.kernel, cfg.apf.dilations, cfg.apf.dropout)
        self.act_emb = ActionEmbedding(self.space, co.d_act)
        self.feedback = SelfMonitoringFeedback(c.n_q, c.vocab, c.frames_per_slot, cfg.feedback.d_f, C, self.tps)
        d_slot = cfg.planner.d_slot
        self.core = DualStateCore(cfg.apf.d_model, cfg.feedback.d_f, co.d_act, d_slot, co.d_h, co.d_a, co.d_i,
                                  co.d_m, co.d_couple, co.slow_every, co.ffn_mult, co.lru_c)
        r = cfg.rif
        if r.enabled:
            self.rif = ReciprocalInfluenceField(co.d_h, co.d_a, co.d_i, co.d_m, co.d_act, r.d_hidden, r.d_head,
                                                r.n_layers, r.horizon_slots, r.n_ensemble, len(r.events),
                                                r.listener_heads)
            self.d_rif = (r.horizon_slots + 2) if r.listener_heads else 2
        else:
            self.rif, self.d_rif = None, 0
        self.policy = InteractionPolicy(co.d_h, co.d_a, co.d_i, co.d_m, max(co.d_a // 2, 64), self.space, cfg.policy)
        pc = cfg.planner
        if pc.enabled:
            probe_out = c.frames_per_slot * pc.probe_codebooks * c.vocab if pc.probe_codebooks else 0
            self.planner = ShortHorizonPlanner(pc.n_slots, d_slot, pc.n_layers, pc.n_heads, co.d_a, co.d_i,
                                               co.d_act, self.d_rif, probe_out)
        else:
            self.planner = DirectSlotHead(d_slot, co.d_a, co.d_i, co.d_act, self.d_rif)
        self.dnag = DirectNeuralAudioGenerator(d_slot, cfg.dnag.d_model, c.n_q, c.vocab, c.frames_per_slot)

    # ------------------------------------------------------------------ utilities
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict:
        out = {}
        for name in ["apf", "act_emb", "feedback", "core", "rif", "policy", "planner", "dnag"]:
            m = getattr(self, name)
            out[name] = 0 if m is None else sum(p.numel() for p in m.parameters())
        out["total"] = self.n_trainable()
        return out

    def rif_summary(self, rif_out: dict | None, a_w: torch.Tensor) -> torch.Tensor | None:
        """Summary of the RIF prediction for the selected action (weights a_w (B, C)); stop-gradient."""
        if rif_out is None:
            return None
        s = ReciprocalInfluenceField.summarize({k: v.detach() for k, v in rif_out.items()})
        w = a_w.detach()
        if "p_vad" in s:
            p_vad = torch.einsum("bc,bcn->bn", w, s["p_vad"])
            extra = torch.stack([(w * s["p_overlap"]).sum(-1), (w * s["cost_std"]).sum(-1)], -1)
            return torch.cat([p_vad, extra], -1)
        return torch.stack([(w * s["cost_mean"]).sum(-1), (w * s["cost_std"]).sum(-1)], -1)

    def decide(self, st: dict, P: torch.Tensor, mode: str = "rif", select: str = "greedy",
               forced: torch.Tensor | None = None, gumbel_tau: float = 1.0) -> dict:
        """One slot decision. mode: 'rif' (score candidates with the RIF) or 'prior' (learned policy only).
        forced (B,) long overrides the selected action (teacher forcing / logged actions)."""
        Y = (st["y_h"], st["y_a"], st["y_i"], st["m"])
        rif_out = self.rif(*Y, self.act_emb.table()) if self.rif is not None else None
        prior = self.policy.prior_logits(*Y)
        rel = self.policy.relevance_logit(st["y_a"], st["m"])
        score = None
        if mode == "rif" and rif_out is not None:
            summ = ReciprocalInfluenceField.summarize(rif_out)
            score = self.policy.score(summ, torch.sigmoid(rel))
        logits = self.policy.logits(score, prior)
        if forced is not None:
            idx = forced
            a_w = F.one_hot(idx, len(self.space)).to(prior.dtype)
        else:
            idx, a_w = self.policy.select(logits, select, gumbel_tau)
        a_e = a_w @ self.act_emb.table()
        P_new, pinfo = self.planner(P, st["y_a"], st["y_i"], a_e, self.rif_summary(rif_out, a_w))
        return {"P": P_new, "idx": idx, "a_w": a_w, "rif": rif_out, "prior": prior, "relevance": rel,
                "score": score, "logits": logits, "pinfo": pinfo}

    # ------------------------------------------------------------------ training forward
    def forward_train(self, batch: dict, action_mode: str = "teacher", decide_mode: str = "rif",
                      gumbel_tau: float = 1.0) -> dict:
        """Teacher-forced (or Gumbel on-policy) unrolled forward over a sequence of S slots.

        batch: mic_mel/ref_mel (B, T, n_mels) or mic_wav/ref_wav (B, T*480); ai_tokens (B, S, F, Q);
               actions (B, S) (action that governed each slot).  T = 2S.
        Returns per-decision tensors with D = S-1 decisions (decision d predicts slot d+1).
        """
        mic_mel = batch["mic_mel"] if "mic_mel" in batch else self.mel(batch["mic_wav"])
        ref_mel = batch["ref_mel"] if "ref_mel" in batch else self.mel(batch["ref_wav"])
        tokens, actions = batch["ai_tokens"], batch["actions"]
        B, S = actions.shape
        T = S * self.tps
        assert mic_mel.shape[1] == T, (mic_mel.shape, T)
        z, _, apf_aux = self.apf(mic_mel, ref_mel)
        p_user = torch.sigmoid(apf_aux["user_vad_logit"]).unsqueeze(-1)
        f = self.feedback(tokens, actions)
        act_tick = self.act_emb(actions).repeat_interleave(self.tps, dim=1)
        st = self.core.init_state(B)
        P = self.planner.init_plan(B)
        rec = {k: [] for k in ["y_h", "y_a", "y_i", "m", "prior", "relevance", "plan", "gate", "revision",
                               "activity", "idx", "a_w", "score"]}
        rif_rec: dict[str, list] = {}
        use_ckpt = self.cfg.train.checkpoint_activations and self.training

        def slot_ticks(st_, P_, s_):
            plan_sum = P_.mean(1)
            for j in range(self.tps):
                t = s_ * self.tps + j
                st_ = self.core.step(st_, z[:, t], p_user[:, t], f[:, t], act_tick[:, t], plan_sum)
            return st_

        for s in range(S):
            st = checkpoint(slot_ticks, st, P, s, use_reentrant=False) if use_ckpt else slot_ticks(st, P, s)
            if s == S - 1:
                break
            forced = actions[:, s + 1] if action_mode == "teacher" else None
            sel = "gumbel" if action_mode == "gumbel" else "greedy"
            out = self.decide(st, P, decide_mode, sel, forced, gumbel_tau)
            P = out["P"]
            for k in ["y_h", "y_a", "y_i", "m"]:
                rec[k].append(st[k])
            rec["prior"].append(out["prior"])
            rec["relevance"].append(out["relevance"])
            rec["plan"].append(P)
            rec["gate"].append(out["pinfo"]["gate"])
            rec["revision"].append(out["pinfo"]["revision"])
            rec["activity"].append(self.planner.activity(P).squeeze(-1))
            rec["idx"].append(out["idx"])
            rec["a_w"].append(out["a_w"])
            if out["score"] is not None:
                rec["score"].append(out["score"])
            if out["rif"] is not None:
                for k, v in out["rif"].items():
                    rif_rec.setdefault(k, []).append(v)

        res = {k: torch.stack(v, 1) for k, v in rec.items() if len(v)}
        res["rif"] = {k: torch.stack(v, 1) for k, v in rif_rec.items()} if rif_rec else None
        res["slots"] = res["plan"][:, :, 0]  # committed slot vectors (B, D, d_slot)
        res["apf"] = apf_aux
        res["mic_mel"] = mic_mel
        res["z"] = z
        return res

    def dnag_logits(self, res: dict, tokens: torch.Tensor, select: torch.Tensor | None = None):
        """DNAG teacher-forced logits for committed slots. tokens (B, S, F, Q).
        select: optional bool mask (B, D) to subsample decisions. Returns logits (N, F, Q, V), targets (N, F, Q)."""
        slots = res["slots"]
        tgt = tokens[:, 1:]
        prev = tokens[:, :-1, -1, :]
        if select is not None:
            slots, tgt, prev = slots[select], tgt[select], prev[select]
        else:
            slots, tgt, prev = slots.flatten(0, 1), tgt.flatten(0, 1), prev.flatten(0, 1)
        return self.dnag(slots, prev, tgt), tgt

    def probe_logits(self, res: dict, select: torch.Tensor | None = None):
        c, pc = self.cfg.codec, self.cfg.planner.probe_codebooks
        plan = res["plan"]
        if select is not None:
            plan = plan[select]
        else:
            plan = plan.flatten(0, 1)
        return plan_probe_logits(self.planner, plan, c.frames_per_slot, pc, c.vocab)
