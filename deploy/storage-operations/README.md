# storage-operations build inputs

Build contexts for the P100 and 3090 llama-server images that add `GET /storage-operations`.

The build flow, the recipes, and the hardware parameters are described in [`deploy/README.md`](../README.md). Read that file first.

The patches (01-23, 90-95, 91-92) are carried in the shared tree as commits, not as loose patch files.
The tree is the single source of truth for the build.

- `deploy/storage-operations/failclosed-b39983005-input-sha256.txt` and `deploy/storage-operations/input-sha256.txt` are history of old builds.
- Build command shape and recipe table: see `deploy/README.md`.
