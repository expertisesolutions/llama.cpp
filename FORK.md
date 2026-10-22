# Expertise fork of llama.cpp: the `expertise-main` branch

This file is about our fork only. It is not part of upstream llama.cpp.

## What `expertise-main` is

`expertise-main` is the official branch of our fork. It plays the same role for us that `master` plays for upstream (`ggml-org/llama.cpp`).

- It holds upstream llama.cpp plus every change we made on top of it.
- It is the only branch we build our servers from.
- It is the only branch that other work is merged into.

The branch is named after its role, not after what it contains or after an upstream version. Its content changes with time. Its role does not.

## Remotes

| Remote | URL | Use |
|---|---|---|
| `fork` | `https://github.com/expertisesolutions/llama.cpp.git` | Our fork. We push here. |
| `origin` | `https://github.com/ggml-org/llama.cpp.git` | Upstream. We only fetch from it. |

Some checkouts call the upstream remote `upstream` and the fork `origin`. Check with `git remote -v` before you push. We have no push rights on `ggml-org`.

`master` in our fork is reserved as a plain mirror of upstream `master`. It may be out of date. Never commit to it and never merge our work into it.

This fork is public. Do not commit host names, IP addresses, credentials, or any private data. Keep deploy details for our own hosts in the private repositories that hold them.

## Rules

1. Every change to llama.cpp that we run ends up in `expertise-main`. A change that is not in `expertise-main` is not part of our llama.cpp.
2. Start new work from `expertise-main`. Use a short-lived branch (for example `fix/slot-save-error`).
3. When the work is done and tested, merge the branch into `expertise-main` and push it to `fork`.
4. Do not build a production server from a feature branch. Build from a commit that is already in `expertise-main`.
5. Do not rewrite the history of `expertise-main` (no force push, no rebase of published commits). The deploy recipes pin commit IDs.
6. Build and run the tests for the code you touched before you merge. For the server, run the tests in `tools/server/tests`.
7. Keep our patches small and separate from upstream code where you can. A smaller patch conflicts less when we take a new upstream version.

## Normal workflow

```sh
git fetch fork
git switch -c fix/my-change fork/expertise-main
# work, commit, test
git switch expertise-main
git pull --ff-only fork expertise-main
git merge --no-ff fix/my-change
git push fork expertise-main
```

## Taking in upstream changes

We follow upstream on purpose, not automatically. Do this when we want a newer upstream version or a specific upstream fix.

```sh
git fetch origin
git switch -c sync/upstream-<date> expertise-main
git merge origin/master        # or a specific upstream commit
# resolve conflicts, build, run the tests
git switch expertise-main
git merge --no-ff sync/upstream-<date>
git push fork expertise-main
```

Merge upstream into our branch. Do not rebase our branch onto upstream.

llama.cpp moves fast. On 2026-10-22 upstream `master` was 1163 commits ahead of our base. The files we change most are `tools/server/server-context.cpp`, `src/models/qwen4exp.cpp`, and the slot save and restore code. Expect conflicts there.

## Where we started

`expertise-main` was created on 2026-10-22 from the branch `storage-operations` (commit `6ed910655`). That branch is built on upstream commit `e70802a01` and carries these patch groups:

- the Qwen4Exp NextN/MTP draft head (model, GGUF tensors, converter);
- fail-closed handling of slot checkpoint storage errors;
- slot storage operations tracking and the `GET /storage-operations` endpoint;
- the packaging inputs for the P100 and 3090 builds.

The older branch name `storage-operations` is kept only so old references still work. Do not add new work to it.

## Not included

The branch `test/pr26004-slot-checkpoints` is an experiment. It is not part of `expertise-main`.
