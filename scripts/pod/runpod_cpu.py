"""Launch and manage a **CPU-only** RunPod instance for the breadth sweep.

The GPU pod this project first tried was a mistake, and a measured one: `tabfm_devices()`
reported only `CPUExecutionProvider` on Linux exactly as on Windows, so the community
`anofox_tabfm` build runs ONNX on the CPU and the card sat idle for 140 minutes. What the sweep
actually wants is uncontended cores.

    python scripts/pod/runpod_cpu.py check                # read-only: what is billing right now
    python scripts/pod/runpod_cpu.py plan                 # read-only: what would be created
    python scripts/pod/runpod_cpu.py create --yes-i-will-pay
    python scripts/pod/runpod_cpu.py gate POD_ID          # read-only: is this host fast enough?
    python scripts/pod/runpod_cpu.py ssh POD_ID
    python scripts/pod/runpod_cpu.py stop POD_ID          # keeps (and keeps billing) the volume
    python scripts/pod/runpod_cpu.py terminate POD_ID --yes-destroy-the-volume

**Gate before you bootstrap.** Placement varies by host and some are unusably slow. The loop:

    create -> gate -> (PASS: bootstrap) or (FAIL: terminate, create again)

Skipping it cost a session here: four consecutive 32-vCPU pods landed on one host at ~0.08 MB/s
and the first sat 29 minutes without cloning a single object. `vcpuCount` must be a power of 2,
so the sizes are 16, 32, 64 -- and asking for a different size is also how you get placed on a
different host.

Everything except `create` and `terminate` is read-only. That split is the point of the file:
following `black_swan/scripts/cloud/runpod_launch.py`, which this is a CPU-shaped port of rather
than a third independent implementation.

**The account is shared.** `check` lists pods belonging to other people. Confirm a pod is yours
before stopping or terminating anything — the names and creation times are the only evidence
available, and a wrong guess destroys someone else's work.

The credential is read at the point of use from `RUNPOD_API_KEY` or `~/.runpod/token.txt`, and
never printed, logged, or interpolated into a command line.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"

# The gate probe. A real artifact this project actually downloads during bootstrap, rather than a
# synthetic speed-test endpoint, so the number measures the path that matters. ~21 MB, which is
# also long enough for the measurement to settle.
DUCKDB_CLI_URL = (
    "https://github.com/duckdb/duckdb/releases/download/v1.5.5/duckdb_cli-linux-amd64.zip"
)

# Ubuntu 22.04 base with CUDA is unnecessary here, but RunPod's plain images are thin on build
# tooling; bootstrap.sh installs what is missing either way, so the cheapest maintained image
# wins. glibc only has to clear manylinux_2_28 for the wheels this project installs.
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# c = compute-optimised. The sweep is arithmetic and ONNX inference, not memory-bound.
CPU_FLAVORS = ["cpu5c", "cpu3c", "cpu5g", "cpu3g"]
VCPU = 16
CONTAINER_DISK_GB = 60
VOLUME_GB = 40
POD_NAME = "duckdb-rocket-cpu-sweep"


def token() -> str:
    env = os.environ.get("RUNPOD_API_KEY")
    if env:
        return env.strip()
    path = pathlib.Path(
        os.environ.get("RUNPOD_TOKEN_FILE", pathlib.Path.home() / ".runpod" / "token.txt")
    )
    if not path.is_file():
        sys.exit(f"no credential: set RUNPOD_API_KEY or put the token in {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        sys.exit(f"{path} is empty")
    return value


def request(path: str, *, method: str = "GET", body: dict | None = None):
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        REST + path,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {token()}",
            "Content-Type": "application/json",
            "User-Agent": "duckdb-rocket-cpu/1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            text = response.read().decode()
            return json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code} on {method} {path}: {exc.read().decode()[:600]}")


def ssh_key() -> str:
    path = pathlib.Path.home() / ".ssh" / "id_ed25519.pub"
    if not path.is_file():
        sys.exit(f"no public key at {path}; a pod without one cannot be reached")
    return path.read_text(encoding="utf-8").strip()


def recipe(args) -> dict:
    return {
        "name": args.name,
        "imageName": args.image,
        "computeType": "CPU",
        "cpuFlavorIds": CPU_FLAVORS,
        "cpuFlavorPriority": "availability",
        "vcpuCount": args.vcpu,
        "containerDiskInGb": args.container_disk,
        "volumeInGb": args.volume,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "interruptible": False,
        "env": {"PUBLIC_KEY": ssh_key()},
    }


def cmd_check(args) -> int:
    pods = request("/pods")
    rows = pods if isinstance(pods, list) else pods.get("data", [])
    if not rows:
        print("no pods -- nothing is billing")
        return 0
    billing = 0
    for pod in rows:
        status = pod.get("desiredStatus") or pod.get("status", "?")
        cost = pod.get("costPerHr", "?")
        kind = "CPU" if pod.get("computeType") == "CPU" else (
            (pod.get("machine") or {}).get("gpuTypeId") or "GPU")
        print(f"  {pod.get('id'):<18} {pod.get('name', ''):<28} {status:<9} {kind:<18} ${cost}/hr")
        if status == "RUNNING":
            billing += 1
    if billing:
        print(f"\n  {billing} RUNNING and billing. The account is shared -- confirm a pod is "
              f"yours before stopping or terminating it.")
    return 0


def cmd_plan(args) -> int:
    print(json.dumps({**recipe(args), "env": {"PUBLIC_KEY": "<your key>"}}, indent=2))
    print(f"\n  CPU pod, {args.vcpu} vCPU, flavors in priority order: {', '.join(CPU_FLAVORS)}")
    print(f"  disk    {args.container_disk} GB container + {args.volume} GB volume")
    print("  cost    CPU pods are billed per vCPU, around $0.035/vCPU/hr -- so 16 vCPU is")
    print("          ~$0.56/hr, only about 25% under the $0.74/hr RTX 6000 Ada this project")
    print("          first used. **Measured, not assumed**: an earlier version of this file")
    print("          claimed 'roughly an order of magnitude cheaper' and that was simply")
    print("          wrong. The reason to prefer CPU here is that the GPU is provably idle")
    print("          (anofox's ONNX Runtime is CPU-only), not that it is dramatically cheaper.")
    print("          Fewer vCPUs is the actual lever on cost.")
    print("\n  `stop` keeps the volume AND keeps billing it. `terminate` destroys it.")
    print("\nto actually create it, re-run with:  create --yes-i-will-pay")
    return 0


def denied_placement(pod: dict) -> str | None:
    """Why this pod should be handed straight back, or None to keep it.

    The lists live in runpod_gpu.py and are IMPORTED rather than copied. black_swan has this same
    check duplicated across thirteen git-ignored wrapper scripts, so it protects whichever script the
    author happened to remember. One definition, two launchers.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import runpod_gpu as rp
    except Exception:  # noqa: BLE001
        return None  # no deny list reachable: keep the pod rather than guess
    machine = str(pod.get("machineId") or "")
    if machine in rp.DENY_MACHINES:
        return f"machine {machine} has had a dead CUDA stack on every pod so far"
    region = rp.pod_region(str(pod.get("id")))
    if region in rp.DENY_REGIONS:
        return f"region {region}, where every pod this account allocated has died"
    return None


def cmd_create(args) -> int:
    if not args.yes_i_will_pay:
        print("REFUSING: this creates a billable pod. Pass --yes-i-will-pay if that is what "
              "you want.", file=sys.stderr)
        return 1
    for attempt in range(1, args.attempts + 1):
        pod = request("/pods", method="POST", body=recipe(args))
        pod_id = pod.get("id") or (pod.get("data") or {}).get("id")
        if not pod_id:
            print(json.dumps(pod, indent=2)[:1500])
            print("REFUSING to continue: create returned no pod id")
            return 1
        reason = None if args.allow_region else denied_placement(pod)
        if reason:
            # Printed before anything that looks like success, so a rejected pod cannot be mistaken
            # for the one to use.
            print(f"attempt {attempt}/{args.attempts}: {pod_id} -- {reason}; "
                  f"terminating and retrying")
            request(f"/pods/{pod_id}", method="DELETE")
            if attempt == args.attempts:
                print(f"\nREFUSING: all {args.attempts} attempts landed somewhere denied. "
                      f"Nothing is billing.")
                return 1
            time.sleep(20)
            continue
        print(json.dumps(pod, indent=2)[:1500])
        print(f"\ncreated {pod_id} -- it is billing now. "
              f"Next: runpod_cpu.py ssh {pod_id}")
        print("Remember to terminate it; a forgotten pod is the expensive failure mode.")
        return 0
    return 1


def cmd_ssh(args) -> int:
    pod = request(f"/pods/{args.pod_id}")
    ip = pod.get("publicIp")
    # portMappings comes back as {"22": 42374} -- a private->public dict, not the list of
    # {privatePort, publicPort} objects the older GraphQL API returned.
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") if isinstance(mappings, dict) else None
    if not ip or not port:
        print(f"pod {args.pod_id} has no public 22/tcp mapping yet "
              f"(status {pod.get('desiredStatus')}); it may still be starting.")
        return 1
    print(f"ssh -p {port} root@{ip}")
    print(f"ssh -p {port} root@{ip} 'bash -s' < scripts/pod/bootstrap.sh")
    return 0


def endpoint(pod_id: str) -> tuple[str, int] | None:
    pod = request(f"/pods/{pod_id}")
    ip = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") if isinstance(mappings, dict) else None
    return (ip, port) if ip and port else None


def cmd_gate(args) -> int:
    """Measure a pod's download speed. Read-only; exits non-zero below the floor.

    Run this BEFORE bootstrapping. `black_swan`'s `pod_runner.py` gates hosts the same way,
    after one host with a healthy `nvidia-smi` and 270 kB/s cost a whole session -- and this
    project then repeated the mistake: four consecutive 32-vCPU pods landed on one host at
    ~0.08 MB/s, and the first sat 29 minutes without cloning a single object before anyone
    thought to measure. A 64-vCPU pod placed elsewhere at 34 MB/s. The rule was written down in
    PLAN.md and not applied, which is why it lives in the tool now rather than in prose.
    """
    where = endpoint(args.pod_id)
    if where is None:
        print(f"pod {args.pod_id} has no public 22/tcp mapping yet; it may still be starting.")
        return 1
    ip, port = where

    # -L matters. The release URL 302-redirects, and without it curl downloads a zero-byte
    # redirect body and reports 0 B/s -- on a perfectly healthy host, indistinguishable from
    # the failure this is meant to catch. That misread cost a pod here.
    probe = (
        'curl -sL -o /dev/null -w "%{speed_download} %{size_download}" --max-time '
        f'{args.timeout} {args.url}'
    )
    cmd = ["ssh", "-n", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", f"root@{ip}", probe]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout + 60)
    except subprocess.TimeoutExpired:
        print(f"  {args.pod_id} at {ip}: probe timed out -- treat as FAIL")
        return 1
    try:
        speed, size = (float(x) for x in proc.stdout.split())
    except ValueError:
        print(f"  {args.pod_id} at {ip}: probe returned {proc.stdout!r} {proc.stderr[:200]!r}")
        return 1

    mb = speed / (1024 * 1024)
    print(f"  pod {args.pod_id} at {ip}: {mb:.2f} MB/s ({size:.0f} bytes)")
    if size == 0:
        print("  FAIL: nothing downloaded. If the URL redirects, the probe needs -L.")
        return 1
    if mb < args.floor:
        print(f"  FAIL: below the {args.floor} MB/s floor. Terminate and create another; "
              f"placement varies by host, and vcpuCount must be a power of 2.")
        return 1
    print(f"  PASS: at or above the {args.floor} MB/s floor.")
    return 0


def cmd_stop(args) -> int:
    request(f"/pods/{args.pod_id}/stop", method="POST")
    print(f"stopped {args.pod_id}. The volume persists AND still bills. "
          f"`terminate` to stop paying entirely.")
    return 0


def cmd_terminate(args) -> int:
    if not args.yes_destroy_the_volume:
        print("REFUSING: terminate destroys the pod volume -- the built extension, the "
              "downloaded weights, anything not copied off. Copy what you want first "
              "(`ssh POD_ID` prints the connection line), then pass "
              "--yes-destroy-the-volume.", file=sys.stderr)
        return 1
    request(f"/pods/{args.pod_id}", method="DELETE")
    print(f"terminated {args.pod_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_recipe_flags(sub):
        sub.add_argument("--name", default=POD_NAME)
        sub.add_argument("--image", default=IMAGE)
        sub.add_argument("--vcpu", type=int, default=VCPU)
        sub.add_argument("--container-disk", type=int, default=CONTAINER_DISK_GB)
        sub.add_argument("--volume", type=int, default=VOLUME_GB)

    subparsers.add_parser("check").set_defaults(func=cmd_check)

    plan = subparsers.add_parser("plan")
    add_recipe_flags(plan)
    plan.set_defaults(func=cmd_plan)

    create = subparsers.add_parser("create")
    add_recipe_flags(create)
    create.add_argument("--yes-i-will-pay", action="store_true")
    create.add_argument("--attempts", type=int, default=6,
                        help="retries when a pod lands in a denied region or on a denied machine")
    create.add_argument("--allow-region", action="store_true",
                        help="keep a pod even if its placement is on the deny list")
    create.set_defaults(func=cmd_create)

    for name, func in (("ssh", cmd_ssh), ("gate", cmd_gate),
                       ("stop", cmd_stop), ("terminate", cmd_terminate)):
        sub = subparsers.add_parser(name)
        sub.add_argument("pod_id")
        if name == "terminate":
            sub.add_argument("--yes-destroy-the-volume", action="store_true")
        if name == "gate":
            sub.add_argument("--floor", type=float, default=5.0, help="MB/s, default 5")
            sub.add_argument("--timeout", type=int, default=60, help="probe seconds")
            sub.add_argument("--url", default=DUCKDB_CLI_URL)
        sub.set_defaults(func=func)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
