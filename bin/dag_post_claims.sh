#!/bin/sh
# DAGMan POST script for the claims node: never fail the workflow. If claim checking produced
# nothing, leave a placeholder so referee/editor jobs still have a file to transfer.
if [ ! -s claim_findings.md ]; then
  printf '# Claim verification\n\nUnavailable: the claim-checking job did not produce results.\n' > claim_findings.md
fi
exit 0
