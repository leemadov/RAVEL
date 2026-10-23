"""Run a list of shell jobs with at most N in parallel (each job: 1 BLAS thread by default).

PyTorch CPU training of this model is dominated by many small sequential ops; on 4 cores, four 1-thread jobs give
~2x the throughput of two 2-thread jobs, and oversubscribing threads is catastrophically slow (~30x).

    python scripts/job_queue.py --jobs scripts/jobs_main.txt --parallel 4
Job file: one "<output_dir> <command...>" per line; jobs whose <output_dir>/final.pt or DONE marker exists are skipped.
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    todo = []
    for line in open(a.jobs):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out, cmd = line.split(" ", 1)
        if os.path.exists(os.path.join(out, "final.pt")) or os.path.exists(os.path.join(out, "DONE")):
            continue
        todo.append((out, cmd))
    running = []
    while todo or running:
        running = [(o, p) for o, p in running if p.poll() is None]
        while todo and len(running) < a.parallel:
            out, cmd = todo.pop(0)
            os.makedirs(out, exist_ok=True)
            log = open(os.path.join(out, "stdout.log"), "a")
            p = subprocess.Popen(f"{cmd} && touch {out}/DONE", shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((out, p))
            print(time.strftime("%H:%M:%S"), "started", out, flush=True)
        time.sleep(10)
    print(time.strftime("%H:%M:%S"), "ALL_JOBS_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
