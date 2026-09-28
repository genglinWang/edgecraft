# Security and tenant-boundary notes

## Data sent to an LLM provider

Synthesis sends the task description, device/runtime facts, dataset filenames,
schema and selected sample values, candidate source code, and execution feedback
to the configured provider. Dataset exploration tools can read bounded portions
of files beneath the supplied dataset root. Use data you are authorized to share
with that provider; review its retention policy before using sensitive data.
Provider credentials belong in the local environment and are excluded from
generated candidate subprocesses. Telemetry records token counts, not prompts.
Set `EDGECRAFT_DISABLE_LLM=1` to block model API calls during local checks.

## Logical control-plane isolation

When `EDGECRAFT_API_TENANT_TOKENS` is configured, the API maps opaque bearer
tokens to validated tenant IDs. List/read/cancel/tree/artifact routes filter by
that authenticated tenant and return 404 for another tenant's object. Workspace
and TrialBank paths use an opaque tenant namespace; pretrained caches, CBR
collections, and JSON stores use the same tenant-scoped derivation. Direct
request fields that accept arbitrary dataset or credential paths are disabled
in token-authenticated multi-tenant mode. Instead, tenant-relative resource
handles are resolved beneath `EDGECRAFT_TENANT_DATA_ROOT` and
`EDGECRAFT_TENANT_CREDENTIAL_ROOT`; canonical-path checks reject traversal and
symlink escape, and credential files must be owner-only. The service applies
one named mechanism profile before task workers start rather than mutating
global settings per request.

Shared components reuse only what their mechanism requires: the scheduler
arbitrates with an opaque tenant key, ready stage, duration estimate, device
requirement, and tree-prior tie-break, while a stage's execution payload stays
bound to its own request; a reviewed calibration export contains sanitized
environment/protocol/graph measurement pairs; and only reproduced, verified
compatibility rules enter shared lookup. User datasets, artifacts, raw failure
observations, prompts, credentials, and private ledgers are not surfaced to or
reused by other tenants through those interfaces. Candidate subprocesses
receive a workspace-local home and cache plus a minimal runtime environment,
excluding controller API keys, tenant-token maps, and SSH-agent state.

With no token mapping, the service is explicitly in single-operator mode. The
CLI binds to loopback by default. Do not bind it to a non-loopback interface
without configuring authentication and an appropriate network boundary.

Verified-rule and calibration exporters are allowlist based. Raw errors,
prompts, tenant or dataset identifiers, generated code, model paths, artifacts,
evidence transcripts, and timestamps are excluded from public snapshots.
Raw failure observations remain tenant-private behind an opaque tenant scope;
only a successfully reproduced, allowlisted rule enters shared lookup.

## Trusted-execution boundary

EdgeCraft is a research prototype. The edge runner executes `run.sh` and sources
`meta.env` from submitted bundles. Treat both as trusted code and operate the
runner only on isolated research devices with a dedicated, least-privilege
account.

Do not commit credentials or connection details. Supply a private key and host
at runtime, restrict key permissions, and use normal SSH host-key management for
your environment. Controller and transfer commands pass that key explicitly
with `IdentitiesOnly=yes`; they do not fall back to another agent identity.
Container tags and native runtime paths also belong in local manifests, not
source code.

The optional systemd installation includes retention cleanup scoped to the
runner's configured state directory. Review its environment file before
enabling the timer on a shared host. Controller-side collection preserves
remote jobs by default.

The logical controls above implement authenticated service access and
separation under a trusted-generated-code assumption. A public production
deployment additionally requires hostile-code isolation, production identity
and secret management, protected storage and networking, and denial-of-service
controls. The research runner should remain on an isolated, access-controlled
network.
