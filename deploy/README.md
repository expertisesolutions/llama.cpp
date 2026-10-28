# Deploy recipes

This directory holds the container recipes for the llama-server images of the P100 and the RTX 3090 hosts.

## Single source tree

The source of every image is the branch `storage-operations`.
Commit: commit X (filled in when pin/prefix-pins is merged into storage-operations).

The P100 image and the 3090 image compile the same tree. The recipes apply no patches: all earlier patches are commits in the branch.

The two builds differ only in build parameters:

| Parameter | P100 | 3090 |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | 60 | 86 |
| `GGML_RPC` | ON | OFF |
| Recipe | `deploy/p100/Dockerfile.cuda-12.p100-from-tree` | `deploy/3090/Containerfile.cuda-sm86.from-tree` |
| Image name | `llama-p100-qwen4-mtp` | `llama-cuda-3090` |

Both recipes use the base image `nvidia/cuda:12.2.2-devel-ubuntu22.04` for the build stage and `nvidia/cuda:12.2.2-runtime-ubuntu22.04` for the runtime stage.

## Build

Each build starts in a new, empty build directory. The recipe reads the source from a file named `src.tar` in the build context.

1. Make sure that the commit you want is the tip of `storage-operations` in your llama.cpp checkout. Write down its full SHA.
2. Create the new build directory and the target directory (`p100` or `3090`) inside it.
3. Export the tree with `git archive`, without a `.git` directory:

       git archive <full-sha> > <new-build-dir>/<target>/src.tar

4. Copy the recipe for the target into `<new-build-dir>/<target>/`.
5. Run the build:

       timeout --signal=TERM --kill-after=120s 6h podman build --layers \
         --label local.source-commit=<full-sha> \
         --label local.source-branch=storage-operations \
         --iidfile <new-build-dir>/<target>.iid \
         -t localhost/<image>:storage-operations-<shortsha> \
         -f <new-build-dir>/<target>/<recipe> <new-build-dir>/<target>

The label `local.source-commit` is the build identity of the image. The build context has no `.git` directory, so `llama-server --version` and the `build_info` field print "unknown". Read the label to find the commit:

    podman image inspect <image> --format '{{ index .Labels "local.source-commit" }}'

## Hardware status

| Host | Image in use | Status |
|---|---|---|
| P100 | built from the single source tree | follows the commit above |
| 3090 | `llama-cuda-3090:storage-operations-6ed910655` | pending |

3090: pending. It still runs `llama-cuda-3090:storage-operations-6ed910655`. It must be rebuilt from the same commit with `deploy/3090/Containerfile.cuda-sm86.from-tree`. No build and no restart of the 3090 was done in the work that added prefix pins.

## Judge instance parameters (P100)

The permission judge runs one `llama-server` per P100 GPU with these parameters:

    -c 32768 -np 1 --kv-unified -ctk f16 -ctv f16 -fa on -b 2048 -ub 512
    --spec-type none --cache-ram 0 --checkpoint-min-step 0 --jinja --slots
    --prefix-pins 8

`--prefix-pins 8` is the value measured in the VRAM test (evidence file `task-19-vram.json`). On the Qwen hybrid model on a P100, each extra sequence costs a fixed 150.5 MiB, and the 8 pins cost 1204 MiB in total. All of that memory is in the recurrent state, which is sized by the number of sequences at startup. The KV cache itself does not grow.

`--prefix-pins 0` turns the feature off. Then the server behaves as before.

Creating a pin prefills it in one call that cannot be cancelled and blocks the server for its duration (about 6 s for a typical pin, up to about 60 s at the default `--prefix-pin-max-tokens`). See "Limitations" in the prefix pins section of `tools/server/README.md`.

See "Prefix pins" in `tools/server/README.md` for the flags, the startup rules, and the HTTP API.

Measurements that need the real model on the P100 test GPU (latency, hit rate, equivalence of results) are to be measured (tasks 9 and 10).

## Older build inputs

`deploy/storage-operations/` keeps the history of earlier builds. See `deploy/storage-operations/README.md`.
