# Synthetic review fixtures

These two JSON files are deliberately tiny, synthetic examples of the
allowlisted formats accepted by the EdgeCraft evidence layer:

- `synthetic_verified_rules.json` shows a reproduced, exact-context hardware
  compatibility rule after private failure text and paths have been removed.
- `synthetic_calibration.json` shows a same-graph P1/P2 measurement pair after
  tenant, dataset, workspace, transcript, and timestamp provenance has been
  removed.

They exist only to make schemas and review commands inspectable. They are not
paper measurements, are not used as default decision evidence, and cannot
reproduce any numeric result in the paper. Real exports are created explicitly
with `edgecraft evidence export-rules` and
`edgecraft evidence export-calibration` after their source records pass the
corresponding verification gates.
