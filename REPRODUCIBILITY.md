# Synthesis behavior

Start with the [tutorial](README.md#tutorial), then use the device and dataset
guides for your environment. The release implements the main synthesis service;
the tests exercise its policies with small examples.

## Search and verification

A request supplies a task, dataset, edge target, runtime, and SLOs. The `paper`
profile permits up to 24 candidates. Each candidate contains data-loading,
training, export, and inference code. The tree retains competing candidates;
verified results guide subsequent selection and revision.

P0 checks static contracts and known incompatibilities. P1 measures an
initialized model with the candidate's deployment graph. It can reject a
candidate only when a conservative bound violates the SLO and prior completed
P1/P2 measurements match the graph, device, runtime, precision, input profile,
and measurement protocol. New contexts therefore proceed to P2. P2 performs
training and device evaluation and is the only stage that can accept an artifact.

Calibration and verified rules are learned during use. A failure becomes a
reusable rule after reproduction in the same environment. Later matches must
satisfy the rule's predicate and scope; changing runtime or environment prevents
reuse of a stale match. The files in `fixtures/` illustrate the formats with
synthetic values and are not loaded automatically.

## Results and costs

Each run has a tenant-scoped workspace and `trial_bank.json`. The bank contains
candidate relationships, stage status, verification evidence, and resource
measurements. Provider-reported input/output tokens are written to the configured
telemetry path. GPU allocation time and edge execution time are recorded
separately. Failed attempts remain part of run costs.

A feasible result satisfies every active SLO with full verification. If the
budget ends first, the CLI reports the best verified result and any remaining
gaps as `completed_infeasible`. A failed pipeline is reported as a failure rather
than a successful deployment.

## Shared service

Use `edgecraft serve --help` and `edgecraft task --help` for the optional service
interface. Configure tenant authentication before exposing a non-loopback
endpoint. The shared scheduler alternates tenants and dispatches ready training
and device work to separate resource pools. See [Security](SECURITY.md) for
the execution model and provider-data disclosure.

`edgecraft profile show paper` prints the active design settings. The
`paper-audit` profile additionally executes would-prune candidates for policy
testing. The `offline` profile lowers search and training budgets; it is not a
replacement for an LLM provider. Set `EDGECRAFT_DISABLE_LLM=1` when running tests
that must never contact a model API.
