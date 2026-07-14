# Releasing pimm

## Versions

Stable releases use `vMAJOR.MINOR.PATCH` tags matching the `version` in `pyproject.toml`, which is the source of truth.
Bump MAJOR for incompatible changes, MINOR for backward-compatible features, PATCH for fixes.

## Cut a release

1. Create the draft and curate its notes:

```bash
gh release create vX.Y.Z --draft --generate-notes --title "pimm vX.Y.Z" --target main
```

2. Open a PR that bumps the `pyproject.toml` version and refreshes the lock (`uv lock`).
3. Merge it to `main`.

Merging the version bump triggers the Release workflow, which verifies the lock and build, creates the annotated `vX.Y.Z` tag on the merged commit, builds the version-tagged containers, and publishes the draft.

## Retry

Re-run the failed jobs, or dispatch Release manually with `tag` set to `vX.Y.Z` and `ref` set to the release commit SHA (or the existing tag).
Never move a published tag; roll back with a new PATCH release.

## Native wheels

Native wheel tags (`pimm-wheels-*`) are an independent stream, rebuilt only when the CUDA/PyTorch/Python/architecture stack changes.
The pimm release contains the six extensions owned by this repository plus `torch-scatter`, `torch-sparse`, and `torch-cluster`.
Their pinned source distributions and hashes live in `.github/scripts/native-wheels.json`; all nine wheels are built on manylinux 2.28 for RHEL 8 compatibility.

Prepare a compatibility-stack change before merging it to `main`:

1. Push the stack manifest and extension-version changes to a branch.
2. Dispatch **Native wheel release** on that branch with `publish=false`.
   This builds, imports, audits, and uploads a draft wheel release plus its manifest.
3. Download the workflow artifact, resolve `uv.lock` against those exact local wheels, then replace the local source URLs with the final release URLs while retaining the generated SHA-256 hashes.
   Push the lock update.
4. Dispatch the workflow again on the updated branch with `publish=true`.
   The workflow downloads the staged draft assets, verifies their checksums and locked hashes, and publishes those exact files as a public prerelease.
5. Merge only after the public wheel URLs resolve.
   The normal Docker workflow can then build and test the new application images without a dependency race.

Application releases never rebuild native wheels.
Once the compatibility stack has passed the application image and GPU test matrix, its GitHub prerelease can be promoted to a normal release.
