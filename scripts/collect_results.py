"""Copy training logs into results/ (checkpoints/ is not versioned) and render markdown tables from results/*.json.

    python scripts/collect_results.py > results/tables.md
"""
import glob
import json
import os
import shutil

import numpy as np

R = "results"


def copy_logs():
    os.makedirs(f"{R}/train_logs", exist_ok=True)
    for f in glob.glob("checkpoints/**/log.jsonl", recursive=True):
        name = os.path.relpath(os.path.dirname(f), "checkpoints").replace("/", "__")
        shutil.copy(f, f"{R}/train_logs/{name}.jsonl")
        meta = os.path.join(os.path.dirname(f), "meta.json")
        if os.path.exists(meta):
            shutil.copy(meta, f"{R}/train_logs/{name}.meta.json")


def read_jsonl(p):
    return [json.loads(x) for x in open(p)]


def stage1_table():
    rows = []
    for tag in ["stage1_ravel", "stage1_baseA"]:
        p = f"{R}/train_logs/{tag}.jsonl"
        if not os.path.exists(p):
            continue
        L = read_jsonl(p)
        ev = [x for x in L if "eval" in x]
        tr = [x for x in L if "step_s" in x]
        meta = json.load(open(f"{R}/train_logs/{tag}.meta.json"))
        for e in ev:
            rows.append((tag, meta["params"], e["step"], e["eval"]["test_ce"], e["eval"]["test_ce_q0"], e["eval"]["test_acc"]))
        if tr:
            rows.append((tag + " (train)", meta["params"], "mean step_s", round(float(np.mean([x["step_s"] for x in tr[1:]])), 2),
                         "max RSS GB", max(x["rss_gb"] for x in tr)))
    out = ["| run | params | step | held-out CE (all codebooks) | CE codebook 0 | token acc |", "|---|---|---|---|---|---|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def closed_loop_table(path):
    if not os.path.exists(path):
        return f"(missing {path})"
    d = json.load(open(path))
    keys = ["cost", "overlap_slots", "gap_slots", "bargein", "turn_abandon", "reprompt", "response_latency_slots",
            "frac_any_speech", "frac_backchannel", "frac_speak", "frac_speak_slow", "frac_speak_fast"]
    out = ["| policy | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, m in d.get("references", {}).items():
        out.append(f"| {name} | " + " | ".join(f"{m[k]:.3f}" if isinstance(m[k], float) else str(m[k]) for k in keys) + " |")
    for name, m in d["models"].items():
        cells = []
        for k in keys:
            v = m[k]
            cells.append(f"{v['mean']:.3f} ± {v['sem_over_seeds']:.3f}" if v.get("sem_over_seeds") is not None else f"{v['mean']:.3f}")
        out.append(f"| **{name}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    copy_logs()
    print("## Stage 1\n")
    print(stage1_table())
    for p in sorted(glob.glob(f"{R}/e*_closed_loop*.json")):
        print(f"\n## {os.path.basename(p)}\n")
        print(closed_loop_table(p))


if __name__ == "__main__":
    main()
