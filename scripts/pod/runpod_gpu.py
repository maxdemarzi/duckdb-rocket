#!/usr/bin/env python3
"""RunPod: enumerate, price, plan and (only when asked twice) launch a GPU pod.

Stdlib only, on purpose. The `runpod` SDK is fine but it is a 60-dependency install and
this file has to be runnable from a bare Python on any machine that has the token --
including a pod, when the pod is the thing diagnosing itself. Two endpoints do everything
here: the REST API (`rest.runpod.io/v1`) for pods and templates, and the older GraphQL API
(`api.runpod.io/graphql`) for prices and stock, which REST does not expose at all.

    python scripts/cloud/runpod_launch.py check          # who am I, what is running, what is it costing
    python scripts/cloud/runpod_launch.py gpus           # every GPU type: on-demand, spot, stock
    python scripts/cloud/runpod_launch.py plan           # the exact create body and $/hr -- NO call
    python scripts/cloud/runpod_launch.py payload        # build the transfer tarball -- NO network
    python scripts/cloud/runpod_launch.py create --yes-i-will-pay
    python scripts/cloud/runpod_launch.py ssh POD_ID     # the scp/ssh lines for that pod
    python scripts/cloud/runpod_launch.py stop POD_ID    # keeps the volume, keeps billing it
    python scripts/cloud/runpod_launch.py terminate POD_ID

**`create` is the only subcommand that spends money and it will not run without
`--yes-i-will-pay`.** Everything else is read-only. That split is the point of the file:
the expensive decision should be one reviewed command, and the twenty things you want to
know *before* making it should cost nothing.

The token is read from `~/.runpod/token.txt` (override with `RUNPOD_TOKEN_FILE`) or from
`RUNPOD_API_KEY`, at the point of use. It is never logged, never echoed, and never written
anywhere. `--verbose` prints request URLs and bodies but the Authorization header is
constructed inside `_request` and never reaches the printer.

Cloudflare rejects the GraphQL endpoint with 403/1010 when no User-Agent is sent, which
reads exactly like a bad key. `_request` always sets one.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
UA = "black_swan-runpod-launch/1"

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

# --------------------------------------------------------------------------- #
# The launch recipe
# --------------------------------------------------------------------------- #

# Ubuntu 22.04 -> glibc 2.35. Every rai-swan Linux wheel is manylinux_2_28 and there is no
# sdist, so 2.28 is the floor and 22.04 clears it with room. PLAN.md Amendment F once said
# 2.39/Ubuntu 24.04; that was wrong and cost an image rebuild that was never needed.
# See doctor.py::check_glibc, which is the enforcement and carries the derivation.
#
# CUDA 12.8 + torch 2.8 matches what vLLM's own wheels are built against, so `pip install
# vllm` does not replace the 3 GB of torch the image already has.
IMAGE = "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204"

# Fallback: the image behind this account's existing `rai-torch` template (id bipz1htcsn).
# Older torch, but it is the one that has actually run on this account.
IMAGE_FALLBACK = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

GPU = "NVIDIA L40S"

# 48 GB, sm_89, PCIe. The alternates are ordered by how well they substitute for an L40S on
# THIS workload (batched vLLM generation first, 7B QLoRA second), not by price:
#   RTX 6000 Ada  48 GB sm_89 -- same silicon generation, near-identical bf16 throughput
#   RTX A6000     48 GB sm_86 -- same VRAM, ~half the bf16 FLOPs, no FP8
#   A40           48 GB sm_86 -- same again, usually the cheapest 48 GB card available
# All three keep the 7B QLoRA lane and the 14B ablation on the table, which a 24 GB card
# does not. gpuTypePriority=availability means RunPod walks this list rather than failing.
GPU_ALTERNATES = ("NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A6000", "NVIDIA A40")

POD_NAME = "black-swan-l40s"

# Container disk holds the image (~20 GB) plus pip: vllm and its deps are another ~10 GB
# and they land in site-packages, not on the volume.
CONTAINER_DISK_GB = 60

# The volume is /workspace and survives `stop`. It holds HF_HOME (Qwen 7B is 15 GB, 1.5B is
# 3 GB), the checkout, the splits, and adapters/checkpoints (16 LoRA checkpoints ~2.5 GB).
# ~40 GB is the working set; 100 leaves room for a second base model and a merged export.
VOLUME_GB = 100

MOUNT = "/workspace"


def create_body(args: argparse.Namespace) -> dict:
    """The exact JSON POSTed to /v1/pods. Built in one place so `plan` cannot drift from
    `create` -- a dry run that prints something other than what would be sent is worse
    than no dry run."""
    body = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": [args.gpu, *([] if args.no_alternates else GPU_ALTERNATES)],
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "cloudType": args.cloud,
        "computeType": "GPU",
        "interruptible": args.spot,
        "containerDiskInGb": args.container_disk,
        "volumeInGb": args.volume,
        "volumeMountPath": MOUNT,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        # 7B QLoRA wants headroom for the dataloader and vLLM wants CPU for tokenisation.
        "minRAMPerGPU": 32,
        "minVCPUPerGPU": 8,
        "env": {},
    }
    key = _ssh_pubkey()
    if key:
        # This account's SSH keys belong to six different RelationalAI people and none of
        # them is this machine's. Passing the key as env is how the one pod that currently
        # runs on this account was made reachable; relying on the account key list would
        # produce a pod nobody here can log into.
        body["env"]["PUBLIC_KEY"] = key
    return body


def _ssh_pubkey() -> str | None:
    path = pathlib.Path(os.environ.get("RUNPOD_SSH_PUBKEY",
                                       pathlib.Path.home() / ".ssh" / "id_ed25519.pub"))
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #

def token() -> str:
    """From the environment or `~/.runpod/token.txt`, read at the point of use.

    Never returned to a printer, never interpolated into a command line. The one place it
    is allowed to appear is the Authorization header built in `_request`.
    """
    env = os.environ.get("RUNPOD_API_KEY")
    if env:
        return env.strip()
    path = pathlib.Path(os.environ.get("RUNPOD_TOKEN_FILE",
                                       pathlib.Path.home() / ".runpod" / "token.txt"))
    if not path.is_file():
        sys.exit(f"no credential: set RUNPOD_API_KEY or put the token in {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        sys.exit(f"{path} is empty")
    return value


def _request(url: str, *, method: str = "GET", body: dict | None = None,
             verbose: bool = False) -> object:
    payload = json.dumps(body).encode() if body is not None else None
    if verbose:
        print(f"  -> {method} {url}" + (f"\n     {json.dumps(body)}" if body else ""),
              file=sys.stderr)
    request = urllib.request.Request(
        url, data=payload, method=method,
        headers={"Authorization": "Bearer " + token(),
                 "Content-Type": "application/json",
                 "User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:600].decode(errors="replace")
        if exc.code in (401, 403) and "1010" in detail:
            detail += ("\n(Cloudflare 1010 is a missing User-Agent, not a bad key -- "
                       "if you see this from this script the UA constant was dropped)")
        sys.exit(f"HTTP {exc.code} from {url}\n{detail}")
    return json.loads(raw) if raw else None


def gql(query: str, *, verbose: bool = False) -> dict:
    result = _request(GRAPHQL, method="POST", body={"query": query}, verbose=verbose)
    if isinstance(result, dict) and result.get("errors"):
        sys.exit("GraphQL error: " + json.dumps(result["errors"])[:600])
    return (result or {}).get("data", {})


# --------------------------------------------------------------------------- #
# read-only
# --------------------------------------------------------------------------- #

def gpu_table(verbose: bool = False) -> list[dict]:
    data = gql("query { gpuTypes { id displayName memoryInGb maxGpuCount "
               "securePrice communityPrice secureSpotPrice communitySpotPrice } }",
               verbose=verbose)
    return [g for g in data.get("gpuTypes", []) if (g.get("memoryInGb") or 0) > 0]


def stock(gpu_id: str, secure: bool, verbose: bool = False) -> dict:
    """`lowestPrice` is the only place stockStatus is exposed, and it is per (type, cloud).

    High/Medium/Low, or null when nothing matching the filters exists at all. Note that
    null here is 'none available right now', not 'no such GPU' -- the type still appears in
    `gpuTypes` with a price.
    """
    data = gql(f'query {{ gpuTypes(input:{{id:"{gpu_id}"}}) {{ lowestPrice('
               f'input:{{gpuCount:1, secureCloud:{"true" if secure else "false"}}}) '
               f"{{ uninterruptablePrice minimumBidPrice stockStatus }} }} }}", verbose=verbose)
    types = data.get("gpuTypes") or []
    return (types[0].get("lowestPrice") or {}) if types else {}


def datacenters_with(gpu_id: str, verbose: bool = False) -> list[str]:
    data = gql("query { dataCenters { id storageSupport gpuAvailability "
               "{ available stockStatus gpuTypeId } } }", verbose=verbose)
    out = []
    for dc in data.get("dataCenters", []):
        for entry in dc.get("gpuAvailability") or []:
            if entry.get("gpuTypeId") == gpu_id and entry.get("available"):
                out.append(f"{dc['id']}({entry.get('stockStatus')}"
                           f"{',netvol' if dc.get('storageSupport') else ''})")
    return out


def cmd_check(args: argparse.Namespace) -> int:
    """Prove the credential works and say what it is already paying for.

    The balance and the live-pod list are deliberately in the same output. A pod someone
    forgot about is the single most expensive thing this account can be doing, and it is
    invisible from a terminal that only ever runs `create`.
    """
    data = gql("query { myself { id clientBalance currentSpendPerHr spendLimit pods "
               "{ id name desiredStatus costPerHr gpuCount machine "
               "{ gpuDisplayName location } runtime { uptimeInSeconds } } } }",
               verbose=args.verbose)
    me = data.get("myself") or {}
    print(f"credential OK -- account {me.get('id')}")
    print(f"  balance ${me.get('clientBalance'):.2f}    "
          f"spend limit ${me.get('spendLimit')}/hr    "
          f"current spend ${me.get('currentSpendPerHr')}/hr")
    pods = me.get("pods") or []
    if not pods:
        print("  no pods -- nothing is billing GPU time")
        return 0
    print(f"  {len(pods)} pod(s):")
    for pod in pods:
        machine = pod.get("machine") or {}
        up = (pod.get("runtime") or {}).get("uptimeInSeconds")
        print(f"    {pod['id']}  {pod['name']:<24} {pod['desiredStatus']:<9} "
              f"{machine.get('gpuDisplayName')} x{pod.get('gpuCount')} "
              f"@ {machine.get('location')}  ${pod.get('costPerHr')}/hr"
              + (f"  up {up // 60} min" if up else ""))
    running = [p for p in pods if p.get("desiredStatus") == "RUNNING"]
    if running:
        print(f"\n  {len(running)} RUNNING and billing. These are not mine to stop -- "
              f"confirm they are yours before terminating anything.")
    return 0


def cmd_gpus(args: argparse.Namespace) -> int:
    rows = gpu_table(args.verbose)
    rows.sort(key=lambda g: (-(g["memoryInGb"]), g["id"]))
    print(f"{'gpu':<34}{'VRAM':>6}  {'sec OD':>7}{'sec spot':>9}"
          f"{'com OD':>8}{'com spot':>9}  max")
    for g in rows:
        if args.min_vram and g["memoryInGb"] < args.min_vram:
            continue
        def money(value):
            return f"${value:.2f}" if value else "-"
        print(f"{g['displayName']:<34}{g['memoryInGb']:>4}GB  "
              f"{money(g['securePrice']):>7}{money(g['secureSpotPrice']):>9}"
              f"{money(g['communityPrice']):>8}{money(g['communitySpotPrice']):>9}"
              f"  {g.get('maxGpuCount')}")
    print("\nstock and datacenters for the target and its alternates:")
    for gpu_id in (GPU, *GPU_ALTERNATES):
        secure, community = stock(gpu_id, True), stock(gpu_id, False)
        print(f"  {gpu_id:<34} secure {secure.get('stockStatus') or 'none':<7} "
              f"community {community.get('stockStatus') or 'none':<7} "
              f"{' '.join(datacenters_with(gpu_id, args.verbose)) or '(no listed DC)'}")
    print("\nA GPU can be rentable with '(no listed DC)': community hosts outside RunPod's\n"
          "own datacenters do not appear in the dataCenters map.")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Everything `create` would do, priced, without doing it."""
    body = create_body(args)
    live = stock(args.gpu, args.cloud == "SECURE", args.verbose)
    if args.spot:
        rate = live.get("minimumBidPrice")
        basis = "spot / interruptible -- can be reclaimed mid-run"
    else:
        rate = live.get("uninterruptablePrice")
        basis = "on-demand"
    # Measured from this account's own billing history: 1200 GB billed for 24 h cost
    # $3.333, i.e. $0.00278/GB/day. Container disk and volume are both billed.
    disk = (args.container_disk + args.volume) * 0.00278 / 24

    print("POST " + REST + "/pods")
    print(json.dumps(body, indent=2))
    print()
    print(f"  GPU     {args.gpu}  ({args.cloud.lower()}, {basis})")
    print(f"          alternates in priority order: "
          f"{', '.join(GPU_ALTERNATES) if not args.no_alternates else '(none)'}")
    print(f"          stock right now: {live.get('stockStatus') or 'NONE AVAILABLE'}")
    print(f"  cost    ${rate}/hr GPU + ~${disk:.3f}/hr disk "
          f"({args.container_disk} GB container + {args.volume} GB volume)")
    if rate:
        print(f"          = ~${rate + disk:.2f}/hr, ~${(rate + disk) * 24:.2f}/day if left running")
    print(f"  ssh key {'from ' + str(pathlib.Path.home() / '.ssh' / 'id_ed25519.pub') if _ssh_pubkey() else 'NONE -- the pod will be unreachable over ssh'}")
    print()
    print("  `stop` keeps the volume and keeps billing it (~$"
          f"{args.volume * 0.00278 / 24:.3f}/hr). `terminate` destroys it.")
    print("\nto actually create it, re-run with:  create --yes-i-will-pay")
    return 0


# --------------------------------------------------------------------------- #
# payload
# --------------------------------------------------------------------------- #

# corpus/ and splits/ come from the public HF dataset on the pod (see runpod_bootstrap.sh),
# so they are NOT in the tarball. These two are the exception: they are Spider inputs that
# evaluate.py needs and that backup_hf.py does not publish, and they are 1.8 MB together.
DATA_EXTRAS = ("corpus/tables.json", "corpus/spider_dev.json")


def cmd_payload(args: argparse.Namespace) -> int:
    """Everything git would keep, plus the two unpublished Spider inputs, as one tar.gz.

    `--cached --others --exclude-standard`, i.e. tracked files AND untracked ones that
    .gitignore does not exclude, read from the WORKING TREE rather than from HEAD.

    Not `git ls-files` alone, and not `git archive HEAD`: both of those omit files that
    have not been committed yet, and the first thing you write before a first launch is a
    launch script. This bootstrap was itself absent from the first payload built here for
    exactly that reason -- the payload silently shipped without the script it exists to
    deliver. .gitignore already excludes `.claude/worktrees/`, `data/`, `*.jsonl`,
    `*.safetensors` and `__pycache__`, so it is also the correct exclusion list.

    `.sh` files are rewritten to LF on the way in. A shell script authored on Windows
    carries `#!/bin/bash\\r` and dies with "bad interpreter" AFTER a transfer that reported
    success -- which is the most confusing possible time to find out.
    """
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, check=True, capture_output=True).stdout
    names = sorted({n.decode() for n in listing.split(b"\0") if n})

    data_root = pathlib.Path(os.environ.get("BLACK_SWAN_DATA",
                                            pathlib.Path.home() / "black_swan_data"))
    missing = [e for e in DATA_EXTRAS if not (data_root / e).is_file()]
    if missing:
        sys.exit(f"missing from {data_root}: {', '.join(missing)}")

    with tarfile.open(out, "w:gz") as tar:
        for name in names:
            source = REPO / name
            if not source.is_file():
                continue
            if name.endswith(".sh"):
                blob = source.read_bytes().replace(b"\r\n", b"\n")
                info = tarfile.TarInfo(f"black_swan/{name}")
                info.size, info.mode = len(blob), 0o755
                import io
                tar.addfile(info, io.BytesIO(blob))
            else:
                tar.add(source, arcname=f"black_swan/{name}")
        for extra in DATA_EXTRAS:
            tar.add(data_root / extra, arcname=f"data/{extra}")

    size = out.stat().st_size
    print(f"{out}  {size / 1e6:.2f} MB  ({len(names)} repo files + "
          f"{len(DATA_EXTRAS)} data files)")
    for required in ("doctor.py", "scripts/cloud/runpod_bootstrap.sh"):
        if required not in names:
            sys.exit(f"payload is missing {required} -- it would be useless on the pod")
    print("\npush it with:")
    print(f"  python {pathlib.Path(__file__).name} ssh POD_ID    # prints the scp line")
    return 0


# --------------------------------------------------------------------------- #
# pod lifecycle
# --------------------------------------------------------------------------- #

def cmd_create(args: argparse.Namespace) -> int:
    if not args.yes_i_will_pay:
        cmd_plan(args)
        print("\nREFUSING: create bills continuously from the moment the pod starts.\n"
              "Pass --yes-i-will-pay if that is what you want.")
        return 1
    existing = gql("query { myself { pods { id name desiredStatus } } }",
                   verbose=args.verbose).get("myself", {}).get("pods") or []
    clash = [p for p in existing if p["name"] == args.name and p["desiredStatus"] == "RUNNING"]
    if clash and not args.allow_duplicate:
        print(f"REFUSING: {clash[0]['id']} is already RUNNING as '{args.name}'. "
              f"Use it, or pass --allow-duplicate.")
        return 1
    pod = _request(REST + "/pods", method="POST", body=create_body(args), verbose=args.verbose)
    print(json.dumps(pod, indent=2))
    print(f"\ncreated {pod.get('id')} -- it is billing now. "
          f"Next: {pathlib.Path(__file__).name} ssh {pod.get('id')}")
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    pod = _request(f"{REST}/pods/{args.pod_id}", verbose=args.verbose)
    ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
    port = ports.get("22")
    if not (ip and port):
        print(f"pod {args.pod_id} has no public 22/tcp mapping yet "
              f"(status {pod.get('desiredStatus')}); it may still be starting.")
        return 1
    print(f"# {pod.get('name')} -- {pod.get('imageName')}  ${pod.get('costPerHr')}/hr")
    print(f"ssh -p {port} root@{ip}")
    print(f"scp -P {port} PAYLOAD.tgz root@{ip}:{MOUNT}/")
    print(f"ssh -p {port} root@{ip} 'cd {MOUNT} && tar xzf PAYLOAD.tgz && "
          f"bash black_swan/scripts/cloud/runpod_bootstrap.sh'")
    print(f"# artefacts back:")
    print(f"scp -P {port} -r root@{ip}:{MOUNT}/black_swan/adapters ./adapters")
    print(f"scp -P {port} root@{ip}:{MOUNT}/black_swan/doctor.json ./doctor.pod.json")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    print(json.dumps(_request(f"{REST}/pods/{args.pod_id}/stop", method="POST",
                              verbose=args.verbose), indent=2))
    print("stopped. The volume persists AND still bills. `terminate` to stop paying entirely.")
    return 0


def cmd_terminate(args: argparse.Namespace) -> int:
    if not args.yes_destroy_the_volume:
        print("REFUSING: terminate destroys the pod volume -- adapters, checkpoints, the HF "
              "cache. Copy anything you want first (`ssh POD_ID` prints the scp lines), "
              "then pass --yes-destroy-the-volume.")
        return 1
    _request(f"{REST}/pods/{args.pod_id}", method="DELETE", verbose=args.verbose)
    print(f"terminated {args.pod_id}")
    return 0


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="print request URLs and bodies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_recipe_flags(sub):
        sub.add_argument("--gpu", default=GPU)
        sub.add_argument("--name", default=POD_NAME)
        sub.add_argument("--image", default=IMAGE,
                         help=f"default {IMAGE}; fallback {IMAGE_FALLBACK}")
        sub.add_argument("--cloud", default="COMMUNITY", choices=("COMMUNITY", "SECURE"))
        sub.add_argument("--spot", action="store_true",
                         help="interruptible; cheaper, and can be reclaimed mid-run")
        sub.add_argument("--container-disk", type=int, default=CONTAINER_DISK_GB)
        sub.add_argument("--volume", type=int, default=VOLUME_GB)
        sub.add_argument("--no-alternates", action="store_true",
                         help="fail rather than substitute another 48 GB card")

    subparsers.add_parser("check").set_defaults(func=cmd_check)

    gpus = subparsers.add_parser("gpus")
    gpus.add_argument("--min-vram", type=int, default=0)
    gpus.set_defaults(func=cmd_gpus)

    plan = subparsers.add_parser("plan")
    add_recipe_flags(plan)
    plan.set_defaults(func=cmd_plan)

    payload = subparsers.add_parser("payload")
    payload.add_argument("--out", default=str(REPO.parent / "black_swan_payload.tgz"))
    payload.set_defaults(func=cmd_payload)

    create = subparsers.add_parser("create")
    add_recipe_flags(create)
    create.add_argument("--yes-i-will-pay", action="store_true")
    create.add_argument("--allow-duplicate", action="store_true")
    create.set_defaults(func=cmd_create)

    for name, func in (("ssh", cmd_ssh), ("stop", cmd_stop), ("terminate", cmd_terminate)):
        sub = subparsers.add_parser(name)
        sub.add_argument("pod_id")
        if name == "terminate":
            sub.add_argument("--yes-destroy-the-volume", action="store_true")
        sub.set_defaults(func=func)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
