# storage-operations build inputs

Build contexts for the P100 and 3090 llama-server images that add `GET /storage-operations`.

- `p100/` and `3090/` are the exact recovered build inputs of the `failclosed-b39983005`
  images (ai-server1 `~/.local/state/llama-builds/failclosed-b39983005`), plus one new
  patch, `patches/p100/95-storage-operations.patch`.
- `failclosed-b39983005-input-sha256.txt` is the original input list of that build, unchanged.
  The recipes and patches 01-94 here match it byte for byte (the XTX recipe is not included).
- `input-sha256.txt` lists every file in `p100/` and `3090/`, including patch 95.
- `95-storage-operations.patch` is `git diff b39983005 20d6c67ad27f4405e44ea20b4e3009551abaf267 -- tools/server`.
- Both recipes check out upstream `1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af` and apply
  `patches/p100/*.patch` in glob order with `patch -p1 -F 0`, so patch 95 is applied after
  94 without a recipe change. CUDA architectures: 60 (P100), 86 (3090).

Build (from a new build directory, never over `failclosed-b39983005`):

    timeout --signal=TERM --kill-after=120s 6h podman build --layers \
      --label local.storage.source-commit=<full-new-sha> \
      --iidfile <new-build-dir>/<target>.iid \
      -t localhost/<image>:storage-operations-<shortsha> \
      -f <new-build-dir>/<target>/<recipe> <new-build-dir>/<target>

| target | image | recipe |
|---|---|---|
| p100 | llama-p100-qwen4-mtp | toolboxes/Dockerfile.cuda-12.qwen4-mtp-p100patches |
| 3090 | llama-cuda-3090 | toolboxes/Containerfile.cuda-sm86.prefill |
