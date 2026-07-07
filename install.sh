#!/bin/sh
# oracle installer — downloads the latest release binary and installs it.
# Usage: curl -fsSL https://raw.githubusercontent.com/efficientsystemsinc/oracle/main/install.sh | sh
set -eu

REPO="${ORACLE_REPO:-efficientsystemsinc/oracle}"

os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
case "$os-$arch" in
darwin-arm64)        target="darwin-arm64" ;;
linux-x86_64|linux-amd64) target="linux-amd64" ;;
*)
    echo "install.sh: no prebuilt binary for $os/$arch — build from source: go build -o oracle ./cmd/oracle" >&2
    exit 1
    ;;
esac

# latest binaries-<version> release tag via the GitHub API (no auth needed on a public repo)
api="https://api.github.com/repos/$REPO/releases"
tag=$(curl -fsSL "$api" | grep -o '"tag_name": *"binaries-[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$tag" ]; then
    echo "install.sh: no binaries-* release found on $REPO" >&2
    exit 1
fi
url="https://github.com/$REPO/releases/download/$tag/oracle-$target.tar.gz"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "downloading $url"
curl -fsSL "$url" | tar -xz -C "$tmp"

# pick the install dir: /usr/local/bin if writable, else ~/.local/bin
dest="/usr/local/bin"
if [ ! -w "$dest" ]; then
    dest="$HOME/.local/bin"
    mkdir -p "$dest"
fi
install -m 0755 "$tmp/oracle" "$dest/oracle"
echo "installed $dest/oracle ($tag)"

case ":$PATH:" in
*":$dest:"*) ;;
*) echo "note: $dest is not on your PATH" ;;
esac

cat <<'EOF'

Next steps:
  1. Configure an LLM (env or ~/.oracle/config):
       ORACLE_LLM_URL=https://api.openai.com/v1/chat/completions
       ORACLE_LLM_KEY=sk-...
       ORACLE_LLM_MODEL=gpt-4.1
     (or Ollama: ORACLE_LLM_URL=http://localhost:11434/v1/chat/completions)
  2. oracle models pull      # local judge + embedder weights (recommended)
  3. oracle init             # create the database
  4. oracle install-daemon   # keep `oracle up` alive (launchd/systemd)
EOF
