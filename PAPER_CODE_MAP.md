# Design and implementation

The `paper` profile connects the three designs through
[`agent/graph.py`](edgecraft/agent/graph.py): propose a candidate, execute its
verification stages, record evidence, and decide whether to continue.

| Design | Implementation | Essential behavior |
| --- | --- | --- |
| Constraint-Aware Synthesis Tree | [Tree](edgecraft/agent/search/refinement_tree.py), [branch decisions](edgecraft/agent/search/branch_judgment.py), [proposals](edgecraft/agent/nodes/proposal_generator.py) | Retains competing branches and their code; LLM selection and revision use verified quality and SLO gaps. A verifier-pruned branch stays closed. |
| Multi-Fidelity Verifier | [Policy](edgecraft/agent/search/verification.py), [execution](edgecraft/agent/nodes/pipeline_executor.py), [calibration](edgecraft/knowledge/calibration_store.py) | P0 checks contracts; P1 measures the deployment graph cheaply and can reject with prior matching calibration; P2 trains and verifies the artifact. Only P2 accepts. |
| Cross-Tenant Shared Services | [Scheduler](edgecraft/scheduler/scheduler.py), [shared runtime](edgecraft/scheduler/shared_runtime.py), [failure rules](edgecraft/knowledge/compatibility/rule_store.py), [reproduction](edgecraft/knowledge/compatibility/repro.py) | Round-robin across tenants, then shortest ready stage with tree priority as a tie-break. Separate GPU and device pools overlap work. Reproduced failures become scoped rules for later candidates. |

The [trial bank](edgecraft/agent/search/trial_bank.py) persists candidates and
measurements. [API resource handling](edgecraft/api/resources.py) and
[tenant workspaces](edgecraft/agent/workspace/manager.py) separate each tenant's
data, credentials, code, and artifacts. [Token telemetry](edgecraft/utils/token_telemetry.py)
and [resource accounting](edgecraft/evaluation/resource_accounting.py) record
costs incurred during execution.

## Tests

- `tests/test_constraint_tree.py`: branch selection, inheritance, and closed branches.
- `tests/test_verifier_policy.py`, `test_calibration_store.py`, and
  `test_p1_construction_guard.py`: decision authority, context matching, and P1 construction.
- `tests/test_scheduler_policy.py`: tenant order, resource allocation, and overlap.
- `tests/test_failure_to_rule.py`: reproduction, exact rule scope, and invalidation.
- `tests/test_profiles_and_tenants.py` and `test_api_tenant_routes.py`: tenant boundaries.
- `tests/test_paper_mechanism_integration.py`: P1 pruning, tree state, and P2 selection
  together, including the requirement that calibration precede the decision.
