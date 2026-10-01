"""Tests for the full experimental fixture and dual-trace execution."""

import copy

import pytest

from agent_a2d.core.enums import InterventionPoint, StageActionType
from agent_a2d.experiment.config import ExperimentConfig
from agent_a2d.experiment.runner import ExperimentRunner
from agent_a2d.provenance.validation import validate_trace
from agent_a2d.tools.registry import DELETE_CUSTOMER_PROFILE


@pytest.fixture
def base_config(hmac_key: bytes) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp_bench",
        trial_id="trial_1",
        is_attack=True,
        raw_payload="[HACK]",
        seed=123,
        hmac_key=hmac_key,
        policy_version="1.0",
        schema_version=1,
        oracle_action_anchor=DELETE_CUSTOMER_PROFILE,
        active_enforcement=None
    )


def test_experiment_reproducibility(base_config: ExperimentConfig):
    """Identical configurations must produce byte-for-byte identical traces."""
    runner = ExperimentRunner()
    crm = {"C123": "active"}
    tool = {"delete_customer": 0}
    
    result1 = runner.run_trial(base_config, crm, tool)
    result2 = runner.run_trial(base_config, crm, tool)
    
    assert result1.reference_trace == result2.reference_trace
    
    evt_timestamps_1 = [e.timestamp for e in result1.reference_trace.events]
    evt_timestamps_2 = [e.timestamp for e in result2.reference_trace.events]
    assert evt_timestamps_1 == evt_timestamps_2
    
    assert result1.reference_trace.initial_snapshot_id == result2.reference_trace.initial_snapshot_id
    assert result1.reference_trace.session_boundary_snapshot_id == result2.reference_trace.session_boundary_snapshot_id
    assert result1.reference_trace.final_snapshot_id == result2.reference_trace.final_snapshot_id


def test_experiment_active_i_stage_and_not_reached_semantics(base_config: ExperimentConfig):
    """Active enforcement at I-stage causes downstream NOT_REACHED.
    
    Crucially, downstream absence explicitly means NOT_REACHED and the trace
    remains completely structurally valid (it is not considered incomplete).
    """
    cfg = copy.copy(base_config)
    object.__setattr__(cfg, "active_enforcement", InterventionPoint.INGESTION)
    
    runner = ExperimentRunner()
    crm = {"C123": "active"}
    tool = {"delete_customer": 0}
    
    result = runner.run_trial(cfg, crm, tool)
    
    assert len(result.reference_trace.events) == 5
    
    assert result.active_trace is not None
    assert len(result.active_trace.events) == 1
    
    evt_i = result.active_trace.events[0]
    assert evt_i.enforced_decision is not None
    assert evt_i.enforced_decision.stage_action == StageActionType.QUARANTINE_INPUT
    assert evt_i.metadata.get("halted") is True
    
    assert len(result.active_trace.sessions[0].events) == 1
    
    assert crm["C123"] == "active"
    
    assert result.reference_trace.initial_snapshot_id == result.active_trace.initial_snapshot_id
    
    # Prove the trace is fully valid despite being short
    validate_trace(result.active_trace)


def test_experiment_active_m_stage(base_config: ExperimentConfig):
    """Active enforcement at M-stage."""
    cfg = copy.copy(base_config)
    object.__setattr__(cfg, "active_enforcement", InterventionPoint.MEMORY_WRITE)
    
    runner = ExperimentRunner()
    result = runner.run_trial(cfg, {"C123": "active"}, {})
    
    assert result.active_trace is not None
    assert len(result.active_trace.events) == 2
    assert result.active_trace.events[1].enforced_decision.stage_action == StageActionType.PREVENT_ACTIVATION
    validate_trace(result.active_trace)


def test_experiment_active_r_stage(base_config: ExperimentConfig):
    """Active enforcement at R-stage."""
    cfg = copy.copy(base_config)
    object.__setattr__(cfg, "active_enforcement", InterventionPoint.RETRIEVAL)
    
    runner = ExperimentRunner()
    result = runner.run_trial(cfg, {"C123": "active"}, {})
    
    assert result.active_trace is not None
    assert len(result.active_trace.events) == 3
    assert result.active_trace.events[2].enforced_decision.stage_action == StageActionType.EXCLUDE_FROM_REASONING
    validate_trace(result.active_trace)


def test_experiment_active_t_stage(base_config: ExperimentConfig):
    """Active enforcement at T-stage."""
    cfg = copy.copy(base_config)
    object.__setattr__(cfg, "active_enforcement", InterventionPoint.TOOL_AUTHORIZATION)
    
    runner = ExperimentRunner()
    result = runner.run_trial(cfg, {"C123": "active"}, {})
    
    assert result.active_trace is not None
    assert len(result.active_trace.events) == 4
    assert result.active_trace.events[3].enforced_decision.stage_action == StageActionType.DENY_TOOL_ACTION
    validate_trace(result.active_trace)
