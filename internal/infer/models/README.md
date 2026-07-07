# models/

Weights are **pinned to the code** by `manifest.json` (committed) and **stored**
as GitHub release assets (no size/bandwidth limits that matter). The binary
self-provisions: any `ORACLE_LOCAL_*` / `ORACLE_MLX` feature lazily downloads
its missing weights, sha256-verifies them against the manifest, and unpacks
into `~/.oracle/models/`. `oracle models` shows status; `oracle models pull`
prefetches. Auth: `GITHUB_TOKEN` or `gh auth login`.

Shipping new weights (atomic with code):
1. `gh release create models-vN <assets>`
2. update `manifest.json` (tag + sha256s) in the same PR as the code that needs them

Old commits keep fetching their own pinned tag — checkout-and-run always gets
the weights that commit was tested with.
