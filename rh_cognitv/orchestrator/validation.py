"""
Validation Pipeline — composable pre-flight checks.

DI-L2-02: Separate from L3 PolicyChain — different scope, different timing.
L2 ValidationPipeline runs *before* the adapter call. L3 PolicyChain runs
*during* execution (retry, timeout, budget enforcement).

Built-in validators:
  - InputSchemaValidator: checks data matches node's expected input (pure L2)
  - DependencyValidator: checks upstream nodes completed in ExecutionDAG (pure L2)
  - BudgetValidator: calls L3 BudgetTracker.can_proceed() as pre-flight check
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rh_cognitv.execution_platform.protocols import BudgetTrackerProtocol

from .models import ValidationContext, ValidationResult
from .nodes import BaseNode
from .protocols import NodeValidatorProtocol, ValidationPipelineProtocol

if TYPE_CHECKING:
    from .execution_dag import ExecutionDAG


# ──────────────────────────────────────────────
# ValidationPipeline
# ──────────────────────────────────────────────


class ValidationPipeline(ValidationPipelineProtocol):
    """Composable chain of validators, run sequentially with short-circuit.

    On first failure the pipeline stops and returns that failure.
    If all validators pass, returns ``ValidationResult.passed()``.
    """

    def __init__(self, validators: list[NodeValidatorProtocol] | None = None) -> None:
        self._validators: list[NodeValidatorProtocol] = list(validators) if validators else []

    def add(self, validator: NodeValidatorProtocol) -> None:
        """Append a validator to the pipeline."""
        self._validators.append(validator)

    @property
    def validators(self) -> list[NodeValidatorProtocol]:
        """Return the validator list (read-only view)."""
        return list(self._validators)

    async def validate(
        self,
        node: BaseNode,
        data: Any,
        context: ValidationContext,
    ) -> ValidationResult:
        for v in self._validators:
            result = await v.validate(node, data, context)
            if not result.ok:
                return result
        return ValidationResult.passed()


# ──────────────────────────────────────────────
# Built-in Validators
# ──────────────────────────────────────────────


class InputSchemaValidator(NodeValidatorProtocol):
    """Validate that the data payload is present when the node requires input.

    For execution nodes that need a prompt (TextNode, DataNode, ToolNode)
    the prompt field is already required by Pydantic. This validator
    focuses on runtime data: if the node declares an ``input_key``
    in ``ext``, the key must be present in ``data`` (when data is a dict).
    """

    async def validate(
        self,
        node: BaseNode,
        data: Any,
        context: ValidationContext,
    ) -> ValidationResult:
        input_key = node.ext.get("input_key") if node.ext else None
        if input_key is not None and isinstance(data, dict):
            if input_key not in data:
                return ValidationResult.failed(
                    f"Missing required input key '{input_key}' in data",
                    validator_name="InputSchemaValidator",
                )
        return ValidationResult.passed()


class DependencyValidator(NodeValidatorProtocol):
    """Validate that all upstream nodes for this node have completed.

    Expects ``context.ext["predecessors"]`` to contain the list of
    predecessor node IDs (provided by the orchestrator from the PlanDAG).
    Checks against ``context.completed_node_ids``.
    """

    async def validate(
        self,
        node: BaseNode,
        data: Any,
        context: ValidationContext,
    ) -> ValidationResult:
        predecessors: list[str] = context.ext.get("predecessors", [])
        missing = [p for p in predecessors if p not in context.completed_node_ids]
        if missing:
            return ValidationResult.failed(
                f"Upstream nodes not yet completed: {', '.join(missing)}",
                validator_name="DependencyValidator",
            )
        return ValidationResult.passed()


class BudgetValidator(NodeValidatorProtocol):
    """Pre-flight budget check using L3's ``BudgetTracker.can_proceed()``.

    Expects ``context.ext["budget_tracker"]`` to hold a ``BudgetTrackerProtocol``.
    If no tracker is present the check is a no-op (passes).
    """

    async def validate(
        self,
        node: BaseNode,
        data: Any,
        context: ValidationContext,
    ) -> ValidationResult:
        tracker: BudgetTrackerProtocol | None = context.ext.get("budget_tracker")
        if tracker is not None and not tracker.can_proceed():
            return ValidationResult.failed(
                "Budget exhausted — cannot proceed",
                validator_name="BudgetValidator",
            )
        return ValidationResult.passed()
