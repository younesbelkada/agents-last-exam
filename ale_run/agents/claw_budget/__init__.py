"""Package marker for the budget-split harness.

Intentionally empty: ale_claw re-exports its deployer here, which drags the
whole vendored harness (and therefore cua-agent + litellm) into any import of
a sibling module. Keeping this bare lets transcript_to_trajectory be imported
and tested on its own.
"""
