#!/bin/sh
# build_release.sh — build the release binary for THIS host and (optionally) publish.
#
# oracle links onnxruntime_go, which requires cgo — so binaries cannot be
# cross-compiled with a plain `go build`; each target is built natively:
#   linux-amd64   build on any linux/amd64 box (or the CI ubuntu runner)
#   darwin-arm64  MUST be built on an Apple Silicon Mac (or the CI macos-14
#                 runner) — see .github/workflows/release.yml, which builds
#                 both on a v* tag and attaches them to the release.
#
# Usage:
#   scripts/build_release.sh <version>            # build dist/oracle-<target>.tar.gz for this host
#   scripts/build_release.sh <version> --publish  # + gh release create/upload binaries-<version>
set -eu

VERSION="${1:?usage: build_release.sh <version> [--publish]}"
PUBLISH="${2:-}"
REPO="${ORACLE_REPO:-efficientsystemsinc/oracle}"

os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
case "$os-$arch" in
darwin-arm64)             target="darwin-arm64" ;;
linux-x86_64|linux-amd64) target="linux-amd64" ;;
*) echo "unsupported host $os/$arch" >&2; exit 1 ;;
esac

mkdir -p "dist/$target"
echo "building $target"
go build -trimpath -ldflags "-s -w" -o "dist/$target/oracle" ./cmd/oracle
tar -czf "dist/oracle-$target.tar.gz" -C "dist/$target" oracle
rm -r "dist/$target"
ls -lh dist/*.tar.gz

if [ "$PUBLISH" = "--publish" ]; then
    gh release create "binaries-$VERSION" "dist/oracle-$target.tar.gz" \
        --repo "$REPO" --title "oracle $VERSION" \
        --notes "oracle $VERSION — install via install.sh" || \
    gh release upload "binaries-$VERSION" "dist/oracle-$target.tar.gz" --repo "$REPO" --clobber
fi
