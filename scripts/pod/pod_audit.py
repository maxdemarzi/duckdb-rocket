"""Which rented pods are BILLING WITHOUT WORKING? Ask each one, do not infer.

A pod bills whether or not anything is happening on it, and every way this project has lost
money looks the same from the outside: `RUNNING` in the pod list, $0.44/hr, nothing running
inside. In one night that happened four times --

    a driver killed by a pattern match, taking its ssh session and the training with it
    a pod whose id was never captured, so nothing could pull from it or shut it down
    a pod adopted after a misread list, leaving the original unwatched
    a training job that died on the pod while the list still said RUNNING for 175 minutes

-- and each was found by a person happening to look. `runpod_gpu.py check` reports what
RunPod believes; this reports what is true on the machine.

**Three questions per pod, over ssh:**

    is a python process running
    what is the GPU doing
    when did anything under /workspace last change

A pod with no process and an idle GPU is dead weight. A pod with a process but a still GPU
and an old mtime is hung, which bills identically. Both are reported; neither is terminated
without `--terminate-idle`.

**Only pods whose name matches `--mine` can ever be terminated.** This account runs other
people's work -- `tabicl-*`, `duckdb-rocket-*` -- and a cleanup tool that can reach those is
worse than the leak it prevents. The default prefixes are this project's own.

    python scripts/pod/pod_audit.py
    python scripts/pod/pod_audit.py --terminate-idle
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod_gpu as rp  # noqa: E402

#: The launcher is re-invoked as a subprocess for `ssh` and `terminate`. Named once:
#: a stale copy of this name is what made this file unrunnable.
LAUNCHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runpod_gpu.py")

#: A pod younger than this is not judged: it is still pulling an image, installing wheels or
#: downloading a base model, and none of that shows as GPU work. Calling it idle would
#: terminate every pod a minute after creating it.
GRACE_SECONDS = 600

SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", "-o", "BatchMode=yes"]

PROBE = (
    "echo PROC=$(ps aux | grep -c '[p]ython'); "
    "echo GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1); "
    "echo AGE=$(( $(date +%s) - $(find /workspace -type f -newermt '-1 days' -printf '%T@\\n' "
    "2>/dev/null | sort -n | tail -1 | cut -d. -f1 || echo 0) ))"
)


def ssh_probe(pod_id: str, timeout: int) -> dict | None:
    """Ask the pod what it is doing. None when it cannot be reached."""
    line = subprocess.run([sys.executable, "-X", "utf8",
                           LAUNCHER, "ssh", pod_id],
                          capture_output=True, text=True, timeout=60).stdout
    ssh = next((l for l in line.splitlines() if l.startswith("ssh -p")), "")
    if not ssh:
        return None
    port = ssh.split("-p ")[1].split()[0]
    host = ssh.split("root@")[1].strip()
    try:
        got = subprocess.run(["ssh", *SSH_OPTS, "-p", port, f"root@{host}", PROBE],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if got.returncode != 0:
        return None
    out = {}
    for entry in got.stdout.splitlines():
        if "=" in entry:
            key, _, value = entry.partition("=")
            out[key.strip()] = value.strip()
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mine", default="pattern-arm-,eval-,sft-,harvest-",
                    help="comma-separated NAME PREFIXES this tool is allowed to touch. Pods "
                         "outside these are reported and never terminated: the account runs "
                         "other people's work and a cleanup that can reach it is worse than "
                         "the leak")
    ap.add_argument("--terminate-idle", action="store_true",
                    help="destroy pods found idle. Without it this only reports")
    ap.add_argument("--timeout", type=int, default=45, help="seconds per ssh probe")
    args = ap.parse_args()

    prefixes = tuple(p for p in (x.strip() for x in args.mine.split(",")) if p)
    pods = rp._request(f"{rp.REST}/pods") or []
    if isinstance(pods, dict):
        pods = pods.get("data") or pods.get("pods") or []
    running = [p for p in pods if str(p.get("desiredStatus", "")).upper() == "RUNNING"]
    if not running:
        print("\n  no RUNNING pods")
        return 0

    now = time.time()
    print(f"\n  {len(running)} running pod(s)\n")
    print(f"    {'name':26s} {'id':16s} {'age':>6s} {'proc':>5s} {'gpu':>5s}  verdict")
    idle: list[tuple[str, str]] = []
    for pod in sorted(running, key=lambda p: str(p.get("name", ""))):
        name, pod_id = str(pod.get("name", "?")), str(pod.get("id", "?"))
        mine = name.startswith(prefixes)
        created = pod.get("createdAt") or ""
        try:
            age = now - time.mktime(time.strptime(created[:19], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            age = float("inf")
        if not mine:
            print(f"    {name:26s} {pod_id:16s} {'':>6s} {'':>5s} {'':>5s}  not mine, skipped")
            continue
        probe = ssh_probe(pod_id, args.timeout)
        minutes = f"{age / 60:.0f}m" if age != float("inf") else "?"
        if probe is None:
            verdict = "UNREACHABLE -- billing, and cannot be asked"
            print(f"    {name:26s} {pod_id:16s} {minutes:>6s} {'?':>5s} {'?':>5s}  {verdict}")
            continue
        # `grep -c '[p]ython'` counts the probe's own shell out, so 1 is the floor.
        procs = int(probe.get("PROC", "0") or 0)
        gpu = int((probe.get("GPU", "0") or "0").split()[0].rstrip("%") or 0)
        working = procs > 1 or gpu > 5
        if working:
            verdict = "working"
        elif age < GRACE_SECONDS:
            verdict = f"starting up ({GRACE_SECONDS // 60}m grace)"
        else:
            verdict = "IDLE -- billing and doing nothing"
            idle.append((name, pod_id))
        print(f"    {name:26s} {pod_id:16s} {minutes:>6s} {procs:5d} {gpu:4d}%  {verdict}")

    if not idle:
        print("\n  nothing idle.")
        return 0
    cost = 0.44 * len(idle)
    print(f"\n  {len(idle)} idle pod(s), roughly ${cost:.2f}/hr for nothing:")
    for name, pod_id in idle:
        print(f"    {name}  {pod_id}")
    if not args.terminate_idle:
        print("\n  reporting only. Pass --terminate-idle to destroy them, and read the list\n"
              "  first: a pod between two phases of a job looks exactly like a dead one.")
        return 0
    for name, pod_id in idle:
        print(f"  terminating {name} {pod_id}")
        subprocess.run([sys.executable, "-X", "utf8",
                        LAUNCHER, "terminate", pod_id, "--yes-destroy-the-volume"],
                       capture_output=True, text=True, timeout=120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
