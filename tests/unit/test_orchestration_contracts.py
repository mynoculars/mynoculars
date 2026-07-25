"""
tests/unit/test_orchestration_contracts.py — orchestration/contracts.py.

Covers ONLY the @validated_worker decorator (D-15): a worker that tries
to return a forbidden top-level key raises WorkerContractViolation
deterministically (a scheduling-order-dependent KeyError elsewhere in
the graph, before this existed), while a worker returning only allowed
keys passes through unchanged.
"""

import pytest

from research_agent.orchestration.contracts import WorkerContractViolation, validated_worker


def test_validated_worker_rejects_illegal_keys():
    @validated_worker
    def bad_worker(payload):
        return {"final_report": "workers must not write this"}

    with pytest.raises(WorkerContractViolation):
        bad_worker(None)


def test_validated_worker_passes_legal_keys():
    @validated_worker
    def good_worker(payload):
        return {"counters": {"search_calls": 1}}

    assert good_worker(None) == {"counters": {"search_calls": 1}}
