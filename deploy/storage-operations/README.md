# storage-operations build inputs

Build contexts for the P100 and 3090 llama-server images that add `GET /storage-operations`.

The patches (01-23, 90-95, 91-92) are now carried in the shared tree as commits, not as loose patch files.
The tree is the single source of truth for the build.

- `deploy/p100/` and `deploy/3090/` contain the from-tree recipes that build from the current tree state.
- `deploy/storage-operations/failclosed-b39983005-input-sha256.txt` and `deploy/storage-operations/input-sha256.txt` are history of old builds.
- CUDA architectures: 60 (P100), 86 (3090).

Build (from a new build directory):

    timeout --signal=TERM --kill-after=120s 6h podman build --layers \
      --label local.source-commit=<full-sha> \
      --label local.source-branch=storage-operations \
      --iidfile <new-build-dir>/<target>.iid \
      -t localhost/<image>:storage-operations-<shortsha> \
      -f <new-build-dir>/<target>/<recipe> <new-build-dir>/<target>

| target | image | recipe |
|---|---|---|
| p100 | llama-p100-qwen4-mtp | Dockerfile.cuda-12.p100-from-tree |
| 3090 | llama-cuda-3090 | Containerfile.cuda-sm86.from-tree |
