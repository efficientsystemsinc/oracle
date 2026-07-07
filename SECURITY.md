# Security

## The data is the sensitive part

oracle's whole purpose is to distill your AI coding sessions into a durable
fact graph. That graph (`~/.oracle/oracle.db`, backups under
`~/.oracle/backups/`) inevitably contains infrastructure details, hostnames,
decisions, and anything else your sessions discussed. Treat `~/.oracle/` like
a credentials directory:

- never commit it, its backups, or probe files mined from it (`oracle
  mineprobes`) to a public repository;
- the daemon binds 127.0.0.1 only — do not reverse-proxy it to a network
  without adding your own auth;
- transcripts are regex-redacted (`redact` in watch.go) before text leaves the
  box, and again on extraction output — but redaction is best-effort, not a
  guarantee.

API keys live in env vars or `~/.oracle/config` (mode 0600 recommended); the
repo tree never contains credentials.

## Reporting

Email security@starling.sh with reproduction details. Please do not open
public issues for exploitable problems.
