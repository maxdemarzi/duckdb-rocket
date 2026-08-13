# RunPod

The L40S lane, end to end. Prices and stock in this file were read from the API on
2026-08-04 and **will drift** — re-read them with `runpod_launch.py gpus` rather than
trusting the table.

```
python scripts/cloud/runpod_launch.py check    # credential + what is already billing
python scripts/cloud/runpod_launch.py gpus     # prices and stock, live
python scripts/cloud/runpod_launch.py plan     # the exact create body and $/hr, no call
```

Everything except `create` is read-only.

## Tooling

| what | where | why there |
|---|---|---|
| `runpodctl` 2.8.1 | `~/.local/bin/runpodctl.exe` | single Go binary, sha256 verified against the release checksums file |
| `runpod` SDK/CLI | `uv tool install runpod` | its own venv; 60 dependencies do not belong in the environment that also holds torch, duckdb and rai-swan |
| `runpod_launch.py` | this repo | **stdlib only** — no install at all, and it runs on a pod that is diagnosing itself |

`runpod_launch.py` needs none of the above. The two installs are for the things it does
not do: `runpodctl send`/`receive` for peer-to-peer file transfer, and the SDK for
serverless work if that ever happens.

**`runpodctl` does not read `RUNPOD_API_KEY`.** With no key configured it prints an empty
pod list and exits 0 — it looks like "you have no pods", not like "I am not authenticated".
Use it only for `send`/`receive`, which are croc-based and need no credential at all;
`runpod_launch.py` covers the authenticated calls.

## The credential

`~/.runpod/token.txt`, read at the point of use, never echoed and never written into this
repo. `RUNPOD_API_KEY` and `RUNPOD_TOKEN_FILE` override.

Two API surfaces, and you need both:

* **REST** `https://rest.runpod.io/v1` — pods, templates, network volumes, billing.
  `/v1/openapi.json` is the authority on the create-pod body.
* **GraphQL** `https://api.runpod.io/graphql` — prices, stock, `myself`. REST exposes none
  of these. It sits behind Cloudflare, which returns **403 with error 1010 when there is no
  User-Agent header** — indistinguishable from a rejected key if you do not know.

## GPUs

Read 2026-08-04. `stock` is RunPod's own High/Medium/Low; `-` means nothing matching was
available in that cloud at that moment, *not* that the type does not exist.

| GPU | VRAM | sm | secure OD | secure spot | community OD | community spot | stock (sec/com) |
|---|---|---|---|---|---|---|---|
| **L40S** | **48 GB** | **89** | **$0.99** | **$0.86** | **$0.79** | **$0.79** | Low / Low |
| RTX 6000 Ada | 48 GB | 89 | $0.84 | $0.77 | $0.74 | $0.74 | Low / Low |
| RTX A6000 | 48 GB | 86 | $0.53 | $0.49 | $0.33 | $0.33 | Medium / — |
| A40 | 48 GB | 86 | $0.44 | $0.44 | $0.35 | $0.30 | High / — |
| L40 | 48 GB | 89 | $0.82 | $0.99 | $0.69 | — | — |
| A100 PCIe | 80 GB | 80 | $1.39 | $1.39 | $1.19 | $1.19 | Low / Low |
| H100 PCIe | 80 GB | 90 | $2.89 | $2.39 | $1.99 | $1.99 | Low / Low |
| RTX PRO 6000 | 96 GB | 120 | $1.99 | $1.89 | $1.69 | — | Low / — |

Datacenters carrying an L40S at that moment: **EU-NL-1 (Low), EUR-IS-2 (Low)**. Neither
supports network volumes, so the pod-local volume is the only persistent disk for this GPU
— which is fine, it survives `stop`, it just cannot be moved between pods.

Availability is *not* limited to that list. A community L40S in **Taiwan** was rented on
this account during this setup; community hosts outside RunPod's own datacenters do not
appear in the `dataCenters` map at all.

### Which alternate, and why not the cheap ones

The alternates in `runpod_launch.py` are ordered by substitutability for *this* workload,
not by price. The card is budgeted for **batched vLLM generation first, training second**
(k=8–16 × 300 questions per iteration), and the 7B QLoRA shipping lane second.

* **RTX 6000 Ada** — same Ada silicon, sm_89, same 48 GB, FP8 available. The true
  substitute, and usually cheaper than an L40S.
* **RTX A6000 / A40** — 48 GB, but **sm_86**: no FP8, roughly half the bf16 throughput.
  They keep 7B QLoRA and the 14B ablation on the table; they will not match L40S generation
  wall-clock. A40 at $0.35 with **High** stock is the one that is actually always there.
* **A100 80 GB** is 1.5× the price for VRAM this lane does not need. 7B *full* fine-tune is
  out of reach on one card regardless — bf16 weights + Adam states + activations is
  70–90 GB — so more VRAM does not unlock a new lane, only a bigger batch.
* 24 GB cards (4090, 5090) are excluded: they drop the 14B ablation and make 7B tight
  again, which is the constraint the L40S was rented to remove.

**Never mix these inside one measured comparison.** The control must be re-run on whichever
card the treatment ran on; `doctor.json` records `gpu`, `vram_gb`, `compute_capability` and
`glibc` so a result is attributable, and it is only useful if you read it.

## Launch

```bash
python scripts/cloud/runpod_launch.py plan                    # review this first
python scripts/cloud/runpod_launch.py create --yes-i-will-pay
```

That is the whole thing. Without `--yes-i-will-pay` it prints the plan and refuses; it also
refuses if a RUNNING pod already has the same name.

**~$0.81/hr** — $0.79 community on-demand L40S + ~$0.019 disk (160 GB at $0.00278/GB/day,
derived from this account's own billing history, not from the price page). **~$19.40/day if
left running.** Secure cloud is $0.99, spot is $0.86 and can be reclaimed mid-run.

What the create body says, and why:

| field | value | reason |
|---|---|---|
| `imageName` | `runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204` | Ubuntu 22.04 → glibc 2.35, above rai-swan's 2.28 floor. cu128 + torch 2.8 matches what vLLM's wheels are built against, so `pip install vllm` does not swap out 3 GB of torch. |
| `gpuTypeIds` | L40S, then the three alternates | `gpuTypePriority: availability` walks the list instead of failing on Low stock. Pass `--no-alternates` when the comparison requires one specific card. |
| `containerDiskInGb` | 60 | image ~20 GB, plus vllm/trl/peft/bitsandbytes in site-packages ~10 GB. |
| `volumeInGb` | 100 | `HF_HOME` (Qwen 7B is 15 GB, 1.5B is 3 GB), the checkout, ~120 MB of splits, 16 LoRA checkpoints ~2.5 GB. Working set is ~40 GB; the rest is a second base model and a merged export. |
| `env.PUBLIC_KEY` | `~/.ssh/id_ed25519.pub` | The account's own key list belongs to six different RelationalAI people and **not** to this machine. Relying on it produces a pod nobody here can log into. |
| `ports` | `22/tcp` | scp in, scp out. Add `8000/http` if you want the vLLM server reachable from outside. |

Fallback image if the newer one misbehaves:
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` — older torch, but it is the
image behind this account's existing `rai-torch` template and it has actually run here.

## Getting the repo and the data across

Three sources, split by size and by what is already published:

```bash
python scripts/cloud/runpod_launch.py payload          # ~1.8 MB tar.gz
python scripts/cloud/runpod_launch.py ssh POD_ID       # prints the scp/ssh lines
```

1. **Repo → tarball → scp.** 300-odd files, 1.8 MB gzipped. Built from
   `git ls-files --cached --others --exclude-standard`, i.e. the working tree including
   files not yet committed. Not `git archive HEAD`: the first payload built here was
   missing `runpod_bootstrap.sh` precisely because the bootstrap had not been committed
   yet. `.sh` files are converted to LF on the way in — a `#!/bin/bash\r` fails with "bad
   interpreter" *after* a transfer that reported success.
2. **Corpus and splits → Hugging Face, pulled pod-side.** `maxdemarzi/black-swan` is public
   and holds `splits/`, `corpus/pyrel-sft-v3.jsonl`, `bird/records.jsonl`,
   `swan_bench/records.jsonl`, `synthesized/`. ~120 MB, and a datacenter-side download beats
   uploading it from a home connection.
3. **`corpus/tables.json` and `corpus/spider_dev.json` → in the tarball.** `evaluate.py`
   needs both and `backup_hf.py` does not publish either. 1.8 MB together, which is why the
   payload is 1.8 MB.

**`bird/databases/` does not go, under any mechanism.** ~1 GB, and `bird/PROVENANCE.json`
records it as a third-party DuckDB conversion with no declared licence. Any BIRD *execution*
lane on a pod needs those rebuilt there, not copied.

Then:

```bash
ssh -p PORT root@IP 'cd /workspace && tar xzf payload.tgz && \
    bash black_swan/scripts/cloud/runpod_bootstrap.sh'
```

`runpod_bootstrap.sh` installs `rai-swan` and the pinned DuckDB, pulls the HF dataset, and
**runs `doctor.py` as its first and last verification step**, then stops. It deliberately
does not launch a run: doctor is the gate, and a script that installs, verifies and trains
in one breath leaves you nowhere to stand when the verification is what failed.

### Artefacts back

```bash
scp -P PORT -r root@IP:/workspace/black_swan/adapters ./adapters
scp -P PORT root@IP:/workspace/black_swan/doctor.json ./doctor.pod.json
scp -P PORT root@IP:/workspace/black_swan/eval_\*.json .
```

Archive `doctor.json` *with* the eval JSON. A number without its `(swan build, DuckDB ABI,
GPU, glibc)` tuple is not attributable, which is the whole reason doctor writes it.

`runpodctl send <path>` on the pod and `runpodctl receive <code>` locally is the alternative
when SSH is awkward — it is croc, peer to peer, and needs no API key on either end.

## glibc: the 2.39 claim was wrong

`doctor.py::check_glibc` gates at **2.28** and is correct. Verified 2026-08-04 against
PyPI: `rai-swan` 0.0.6 publishes `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`,
two macOS wheels, `win_amd64`, `win_arm64` — **and no sdist**. `tests/test_doctor.py`
asserts 2.35 (Ubuntu 22.04) and 2.28 pass and 2.17 (CentOS 7) fails the gate.

The old 2.39 belief came from `rai-swan` 0.0.1, which shipped `manylinux_2_39` variants
*in addition to* 2.28 — pip would have chosen 2.28 anyway. Ubuntu 22.04 is fine and **no
custom image is needed**. Amendment F in `PLAN.md` carried the wrong claim until
2026-08-04 and has been corrected in place.

No sdist means a host genuinely below 2.28 cannot build from source either; there the
remedy really is a newer base image.

## Money

* `check` prints balance, spend limit and every live pod, together. A forgotten pod is the
  most expensive thing this account can do and it is invisible from a terminal that only
  ever runs `create`. **A running L40S at $0.79/hr is $19/day and $570/month.**
* `stop` keeps the volume **and keeps billing it** — ~$0.012/hr for 100 GB. It is not free
  parking; it is cheap parking.
* `terminate` destroys the volume. `runpod_launch.py terminate` refuses without
  `--yes-destroy-the-volume`.
* This account is shared: six SSH keys from different RelationalAI people are registered on
  it, and pods belonging to other people appear in `check`. **Confirm a pod is yours before
  stopping it.**
* Spend limit is $80/hr, which is not a brake at this scale. The brake is `check`.

## Not verified without a live pod

Everything above the `create` line is verified against the live API. These are not:

* that `pip install vllm` on the chosen image resolves without replacing torch;
* the image's Python version (the tag does not say, unlike the older `-py3.11-` tags);
* that `rai-swan` imports and its DuckDB extension loads on that image — the wheel tag says
  it must, `doctor.py` is what proves it;
* actual generation throughput, and therefore whether the L40S is worth $0.79/hr over an
  A40 at $0.35 for this workload.
