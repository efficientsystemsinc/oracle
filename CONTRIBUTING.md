# Contributing

- One package, one file per concern; keep it that way.
- `go vet ./...` and `go test ./...` must pass; CI also runs the selfeval
  regression gate against the synthetic fixture.
- Fail loud: no fallback providers, no empty-list-on-error, no degraded modes.
- Never commit real session content, probe files mined from your own graph,
  or credentials — CI-visible test data must come from
  `scripts/make_synthetic_fixture.py`.
- Small PRs with a clear invariant argument beat big ones.

By contributing you agree your contributions are licensed under the MIT license.
