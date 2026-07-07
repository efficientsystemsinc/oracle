# Publish runbook — taking oracle public

Exact steps for the session that publishes. The `oss-prep` branch tree is the
publishable tree; history is NOT — it contains keys.env and private eval data,
so publish a squashed single commit.

## 0. Preconditions

- You are on the scrubbed branch (`oss-prep`) with a clean `git status`.
- `go vet ./... && go test ./...` green.
- `gitleaks detect --source . --no-git` clean (brew install gitleaks).

## 1. Rotate the Azure key (the old one lived in git history)

```sh
az cognitiveservices account keys renew -n engram-eval-oai -g engram-eval --key-name key1
```

⚠️ This immediately invalidates the old key everywhere. Update
`~/.oracle/config` on this machine (both `ORACLE_LLM_KEY` and
`ORACLE_EMBED_KEY`) with the new value, and any box that still uses it
(ml/ training boxes read `ORACLE_LLM_KEY` from their env now). If the Sweden
account key was ever used (`azure_sweden.key` on training boxes), renew that
too: `az cognitiveservices account keys renew -n engram-eval-oai-sweden -g engram-eval --key-name key1`.

## 2. Create the fresh public repo

```sh
gh repo create efficientsystemsinc/oracle-public --public --description "Bi-temporal fact graph over your AI coding sessions — persistent memory for coding agents"
```

(or reuse the `oracle` name after renaming the private repo; the private repo
must keep its history PRIVATE either way.)

## 3. Push a squashed single-commit tree

```sh
cd ~/code/oracle && git checkout oss-prep
git checkout --orphan public-main
git add -A
git commit -m "oracle: initial public release"
git remote add public git@github.com:efficientsystemsinc/oracle-public.git
git push public public-main:main
```

Verify on GitHub: exactly ONE commit, no keys.env anywhere.

## 4. Re-upload model release assets

`internal/infer/models/manifest.json` pins tag `models-v1` on `efficientsystemsinc/oracle`.
The public repo needs the same release with the same assets (sha256s are
pinned, so upload the identical files):

```sh
# fetch from the private release, re-upload to public
gh release download models-v1 --repo efficientsystemsinc/oracle -D /tmp/oracle-models
gh release create models-v1 --repo efficientsystemsinc/oracle-public \
  /tmp/oracle-models/judge_v2_onnx.tgz \
  /tmp/oracle-models/embedder_v3_onnx.tgz \
  /tmp/oracle-models/mlx_lib.tgz \
  --title "oracle local-model weights v1" --notes "judge_v2 + embedder_v3 ONNX, MLX dylib bundle"
```

Then update `internal/infer/models/manifest.json` `repo` field to the public repo slug (and
`install.sh` / `build_release.sh` `ORACLE_REPO` default if the name differs)
in a follow-up commit. NOTE: the ask-model checkpoints (policy/synth) are not
on any release yet — `scripts/ask_servers.sh` documents that gap.

## 5. First binaries release

```sh
git tag v0.1.0 && git push public v0.1.0   # release.yml builds both targets
# or by hand on each platform: scripts/build_release.sh 0.1.0 --publish
```

Then verify install end-to-end on a clean box:
`curl -fsSL https://raw.githubusercontent.com/efficientsystemsinc/oracle-public/main/install.sh | sh`

## 6. Final verification

```sh
gitleaks detect --source . --no-git          # on the pushed tree
gh api repos/efficientsystemsinc/oracle-public/commits | jq length   # == 1
grep -r "engram-eval\|6vxf2" . --exclude-dir=.git --exclude=publish_runbook.md  # only this runbook's rotate command may name the resource
```

gitleaks result on the scrubbed tree at prep time (2026-07-08):
`no leaks found` (see PR description for the run log). The only intentional
mention of the Azure resource name in the tree is this runbook's rotation
command.

## 7. Local machine follow-ups (non-blocking)

- The launchd label changed `com.sam.oracle` → `com.oracle.daemon`: run
  `launchctl unload ~/Library/LaunchAgents/com.sam.oracle.plist && rm` the old
  plist, then `oracle install-daemon` once the new binary is installed.
- Private eval assets (real `probes*.tsv`, `ask_confidence_probes.json`,
  `docs/judge_audit.md`, `docs/state_2026-07-05.md`) remain in the PRIVATE
  repo's main branch / history and are gitignored going forward; keep using
  them locally via `eval/eval.sh <file>`.
