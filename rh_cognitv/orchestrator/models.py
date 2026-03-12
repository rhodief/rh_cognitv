"""
Orchestrator data models.

Pydantic models for node results, validation, execution tracking,
and orchestrator configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rh_cognitv.execution_platform.types import ID, Timestamp, generate_ulid, now_timestamp
from rh_cognitv.execution_platform.models import (
    ExecutionResult,
    ResultMetadata,
    TokenUsage,
)


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class NodeExecutionStatus(str, Enum):
    """Status of a single node execution in the ExecutionDAG."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"
    WAITING = "waiting"  # escalation in progress


class DAGRunStatus(str, Enum):
    """Overall status of a DAG run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


# ──────────────────────────────────────────────
# NodeResult — uniform output wrapper
# ──────────────────────────────────────────────


class NodeResult(BaseModel):
    """Uniform result produced by every node execution (adapter output)."""

    ok: bool
    value: Any = None
    error_message: str | None = None
    error_category: str | None = None  # maps to L3 ErrorCategory values
    token_usage: TokenUsage | None = None
    metadata: ResultMetadata | None = None

    @classmethod
    def from_execution_result(cls, result: ExecutionResult[Any]) -> NodeResult:
        """
        Normalise an L3 ExecutionResult[T] into a NodeResult.

        Extracts token_usage from T when available (LLMResultData, ToolResultData).
        """
        from rh_cognitv.execution_platform.models import (
            LLMResultData,
            FunctionResultData,
            ToolResultData,
        )

        value: Any = None
        token_usage: TokenUsage | None = None

        if result.value is not None:
            v = result.value
            if isinstance(v, LLMResultData):
                value = v.text
                token_usage = v.token_usage
            elif isinstance(v, FunctionResultData):
                value = v.return_value
            elif isinstance(v, ToolResultData):
                value = {
                    "llm_text": v.llm_result.text,
                    "function_return": v.function_result.return_value,
                }
                token_usage = v.llm_result.token_usage
            else:
                value = v

        return cls(
            ok=result.ok,
            value=value,
            error_message=result.error_message,
            error_category=result.error_category,
            token_usage=token_usage,
            metadata=result.metadata,
        )

    @classmethod
    def success(cls, value: Any = None, **kwargs: Any) -> NodeResult:
        """Convenience factory for a successful result."""
        return cls(ok=True, value=value, **kwargs)

    @classmethod
    def failure(
        cls,
        error_message: str,
        error_category: str | None = None,
        **kwargs: Any,
    ) -> NodeResult:
        """Convenience factory for a failed result."""
        return cls(
            ok=False,
            error_message=error_message,
            error_category=error_category,
            **kwargs,
        )


# ──────────────────────────────────────────────
# ValidationResult / ValidationContext
# ──────────────────────────────────────────────


class ValidationResult(BaseModel):
    """Outcome of a single validation check or the full pipeline."""

    ok: bool
    error_message: str | None = None
    validator_name: str | None = None

    @classmethod
    def passed(cls) -> ValidationResult:
        return cls(ok=True)

    @classmethod
    def failed(cls, message: str, *, validator_name: str | None = None) -> ValidationResult:
        return cls(ok=False, error_message=message, validator_name=validator_name)


class ValidationContext(BaseModel):
    """
    Contextual information passed to validators.

    Carries references needed by validators (e.g. completed node IDs,
    budget tracker) without coupling them to the full orchestrator.
    """

    completed_node_ids: set[str] = Field(default_factory=set)
    ext: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# FlowResult
# ──────────────────────────────────────────────


class FlowResult(BaseModel):
    """
    Outcome of a FlowHandler execution.

    Describes how to mutate DAG traversal state (expand nodes,
    skip branches, redirect, etc.).
    """

    ok: bool = True
    expanded_node_ids: list[str] = Field(default_factory=list)
    skipped_node_ids: list[str] = Field(default_factory=list)
    redirect_to: str | None = None
    data: Any = None  # transformed data to pass downstream
    error_message: str | None = None


# ──────────────────────────────────────────────
# ExecutionDAGEntry
# ──────────────────────────────────────────────


class ExecutionDAGEntry(BaseModel):
    """A single entry in the append-only ExecutionDAG."""

    id: ID = Field(default_factory=generate_ulid)
    node_id: str
    plan_node_ref: str  # which PlanDAG node this came from
    status: NodeExecutionStatus = NodeExecutionStatus.PENDING
    result: NodeResult | None = None
    started_at: Timestamp = Field(default_factory=now_timestamp)
    completed_at: Timestamp | None = None
    parent_entry_id: ID | None = None  # for branching (undo creates a new branch)
    state_version: int | None = None  # L3 ExecutionState snapshot version


# ──────────────────────────────────────────────
# OrchestratorConfig
# ──────────────────────────────────────────────


class OrchestratorConfig(BaseModel):
    """Orchestrator-wide defaults for execution policies."""

    default_timeout_seconds: float = 30.0
    default_max_retries: int = 3
    default_retry_base_delay: float = 0.1
