# Deploy recipes

This directory holds the container recipes for the llama-server images of the P100 and the RTX 3090 hosts.

## Single source tree

The source of every image is the branch `expertise-main`.
Its base is Kmic-68 `p100-optimizations`, commit `7461904d` (upstream `d327bf1f0`, b11869, 2026-10-24). Our patches sit on top of that base.
The older branch `storage-operations` is not a build source anymore. See "Older build inputs".

The P100 image and the 3090 image compile the same tree. The recipes apply no patches: all patches are commits in the branch.

The two builds differ only in build parameters:

| Parameter | P100 | 3090 |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | 60 | 86 |
| `GGML_RPC` | ON | OFF |
| `GGML_CUDA_NCCL` | OFF | default |
| `GGML_CUDA_FA_QUANTS` | all | default |
| `LLAMA_BUILD_TESTS` | ON | ON |
| Recipe | `deploy/p100/Dockerfile.cuda-12.p100-from-tree` | `deploy/3090/Containerfile.cuda-sm86.from-tree` |
| Image name | `llama-p100-qwen4-mtp` | `llama-cuda-3090` |

The P100 flags `GGML_CUDA_NCCL=OFF` and `GGML_CUDA_FA_QUANTS=all` are the same flags that the Kmic build uses. Both options exist in the CMake files of this tree.

`LLAMA_BUILD_TESTS=ON` builds `test-backend-ops`. The recipe runs `cmake --install`, and the runtime stage copies all of `/usr/local`, so `test-backend-ops` is in `/usr/local/bin` of both images. Use it to test the CUDA kernels on the real GPU, for example `test-backend-ops test -o MUL_MAT -b CUDA0`.

Both recipes use the base image `nvidia/cuda:12.2.2-devel-ubuntu22.04` for the build stage and `nvidia/cuda:12.2.2-runtime-ubuntu22.04` for the runtime stage.

## Build

Each build starts in a new, empty build directory. The recipe reads the source from a file named `src.tar` in the build context.

1. Make sure that the commit you want is the tip of `expertise-main` in your llama.cpp checkout. Write down its full SHA.
2. Create the new build directory and the target directory (`p100` or `3090`) inside it.
3. Export the tree with `git archive`, without a `.git` directory:

       git archive <full-sha> > <new-build-dir>/<target>/src.tar

4. Copy the recipe for the target into `<new-build-dir>/<target>/`.
5. Run the build:

       timeout --signal=TERM --kill-after=120s 6h podman build --layers \
         --label local.source-commit=<full-sha> \
         --label local.source-branch=expertise-main \
         --iidfile <new-build-dir>/<target>.iid \
         -t localhost/<image>:expertise-main-<shortsha> \
         -f <new-build-dir>/<target>/<recipe> <new-build-dir>/<target>

The label `local.source-commit` is the build identity of the image. The build context has no `.git` directory, so `llama-server --version` and the `build_info` field print "unknown". Read the label to find the commit:

    podman image inspect <image> --format '{{ index .Labels "local.source-commit" }}'

## Hardware status

All rows run images of the same commit `eb5d302ac` (Kmic base plus the kept patches; see "Single source tree" above).

| Host | Image in use | Status |
|---|---|---|
| P100 Qwen (8081/8082/8083) | `localhost/llama-p100-qwen4-mtp:expertise-main-eb5d302ac` | 8081 in container `llama-p100-qwen`; 8082/8083 in `llama-p100-kmic-eb5d302ac` |
| 3090 prefill (9101) | `localhost/llama-cuda-3090:expertise-main-eb5d302ac` | container `llama-3090-prefill` |
| P100 judge (8183/8185/8187) | `localhost/llama-p100-qwen4-mtp:expertise-main-eb5d302ac` | container `llama-p100-single-kmic-eb5d302ac` |

The 3090 also runs with the runtime workarounds `LLAMA_KQ_MASK_COMPACT=0` and `GGML_CUDA_GDN_GATHER=0` (Kmic bugs not fixed in this commit); the P100 containers run without them.

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

Before the Kmic base, the images were built from the branch `storage-operations`. `deploy/storage-operations/` keeps the history of those builds. See `deploy/storage-operations/README.md`.
