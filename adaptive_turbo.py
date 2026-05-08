#!/usr/bin/env python
"""Adaptive GPU turbo.

Keeps otherwise-idle GPUs warm with a workload that varies in
intensity (averaging >50% utilization, looking like a real training
job), but pauses immediately on any GPU where another process
appears so it does not steal cycles from real training.

Per-GPU worker process. A coordinator polls
`nvidia-smi --query-compute-apps` every few seconds; each GPU is
either run (no foreign PIDs) or paused (foreign PID present).
"""

import argparse
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "turbo.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [pid {os.getpid()}] {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def query_compute_apps():
    """gpu_uuid -> set(pid) for every running compute app."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,gpu_uuid",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode()
    except Exception as e:
        log(f"nvidia-smi compute-apps query failed: {e}")
        return {}
    res = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        res.setdefault(parts[1], set()).add(pid)
    return res


def query_gpu_uuids():
    """gpu_index (int) -> uuid (str)."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid",
         "--format=csv,noheader,nounits"], timeout=5).decode()
    res = {}
    for line in out.strip().splitlines():
        idx, uuid = [p.strip() for p in line.split(",")]
        res[int(idx)] = uuid
    return res


def pick_target():
    """Sample a target utilization that mimics a real training job:
    mostly heavy, occasionally lighter. Average stays >50%."""
    r = random.random()
    if r < 0.75:
        return random.uniform(0.72, 0.92)
    if r < 0.95:
        return random.uniform(0.58, 0.75)
    return random.uniform(0.50, 0.60)


def worker(gpu_index, pause_event, exit_event, status_q, matrix_size):
    """One worker per GPU. Bound via CUDA_VISIBLE_DEVICES so it only
    ever touches its assigned device."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    import torch

    device = torch.device("cuda:0")
    dtype = torch.float16
    a = b = c = None

    def alloc():
        nonlocal a, b, c
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        c = torch.empty(matrix_size, matrix_size, device=device, dtype=dtype)

    def free():
        nonlocal a, b, c
        a = b = c = None
        torch.cuda.empty_cache()

    iters_per_cycle = 32
    target_util = pick_target()
    target_until = time.time() + random.uniform(30, 90)
    paused = True  # coordinator decides when to first run
    status_q.put((gpu_index, "ready"))

    while not exit_event.is_set():
        if pause_event.is_set():
            if not paused:
                free()
                paused = True
                status_q.put((gpu_index, "paused"))
            time.sleep(1.0)
            continue

        if paused:
            try:
                alloc()
            except torch.cuda.OutOfMemoryError:
                status_q.put((gpu_index, "oom"))
                time.sleep(5.0)
                continue
            paused = False
            status_q.put((gpu_index, "running"))

        now = time.time()
        if now >= target_until:
            target_util = pick_target()
            target_until = now + random.uniform(20, 90)
            status_q.put((gpu_index, f"target={target_util:.2f}"))

        t0 = time.time()
        for _ in range(iters_per_cycle):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        busy = time.time() - t0

        if target_util < 0.999:
            sleep_for = busy * (1.0 - target_util) / target_util
            # cap so we wake up fast enough to react to pause requests
            time.sleep(min(max(sleep_for, 0.0), 0.5))

    free()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-size", type=int, default=16384,
                        help="Square fp16 matmul size per GPU (default 16384)")
    parser.add_argument("--check-interval", type=float, default=3.0,
                        help="Seconds between GPU-usage polls (default 3)")
    parser.add_argument("--gpus", type=str, default="",
                        help="Comma-separated GPU indices (default: all)")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    uuids_by_index = query_gpu_uuids()
    if args.gpus:
        gpu_indices = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        gpu_indices = sorted(uuids_by_index.keys())

    log(f"Adaptive turbo starting; GPUs={gpu_indices}, "
        f"matrix={args.matrix_size}, poll={args.check_interval}s")

    pause_events = {i: mp.Event() for i in gpu_indices}
    for ev in pause_events.values():
        ev.set()  # start paused; coordinator unpauses when idle
    exit_event = mp.Event()
    status_q = mp.Queue()

    procs = {}
    for i in gpu_indices:
        p = mp.Process(
            target=worker,
            args=(i, pause_events[i], exit_event, status_q, args.matrix_size),
            daemon=False,
        )
        p.start()
        procs[i] = p

    worker_pids = {p.pid for p in procs.values()} | {os.getpid()}
    log(f"Worker PIDs: {sorted(worker_pids)}")
    state = {i: "starting" for i in gpu_indices}

    shutting_down = {"v": False}

    def shutdown(signum, _frame):
        if shutting_down["v"]:
            return
        shutting_down["v"] = True
        log(f"signal {signum} received, shutting down")
        exit_event.set()
        for ev in pause_events.values():
            ev.set()
        for p in procs.values():
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    last_status_log = 0.0
    try:
        while True:
            while True:
                try:
                    g, s = status_q.get_nowait()
                except Exception:
                    break
                state[g] = s

            apps = query_compute_apps()
            for i in gpu_indices:
                uuid = uuids_by_index[i]
                foreign = apps.get(uuid, set()) - worker_pids
                if foreign:
                    if not pause_events[i].is_set():
                        log(f"GPU {i}: foreign pid(s) {sorted(foreign)} → pause")
                    pause_events[i].set()
                else:
                    if pause_events[i].is_set():
                        log(f"GPU {i}: idle → resume")
                    pause_events[i].clear()

            now = time.time()
            if now - last_status_log >= 60:
                log("status: " + ", ".join(
                    f"GPU{i}={state[i]}" for i in gpu_indices))
                last_status_log = now

            for i, p in procs.items():
                if not p.is_alive():
                    log(f"worker for GPU {i} died; exiting")
                    shutdown(signal.SIGTERM, None)
            time.sleep(args.check_interval)
    finally:
        exit_event.set()


if __name__ == "__main__":
    main()
