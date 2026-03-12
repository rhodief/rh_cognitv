"""
DAGOrchestrator — core traversal engine (Stage 1: sequential).

DD-L2-03: Topological Sort + Ready Queue.
DI-L2-06: Interrupt check per-node.

The orchestrator:
1. Takes a frozen PlanDAG and data
2. Iterates through nodes in topological order via a ready queue
3. For each node: validate → execute via adapter → snapshot state → record
4. Returns the completed ExecutionDAG
"""

from __future__ import annotations

from typing import Any

from rh_cognitv.execution_platform.errors import InterruptError
from rh_cognitv.execution_platform.protocols import ExecutionStateProtocol

from .adapters import AdapterRegistry, PlatformRef
from .execution_dag import ExecutionDAG
from .models import (
    DAGRunStatus,
    NodeExecutionStatus,
    NodeResult,
    OrchestratorConfig,
    ValidationContext,
)
from .nodes import BaseNode, FlowNode
from .plan_dag import PlanDAG
from .protocols import OrchestratorProtocol
from .validation import ValidationPipeline


class DAGOrchestrator(OrchestratorProtocol):
    """Sequential ready-queue DAG traversal engine.

    Args:
        adapter_registry: Registry mapping node kinds to adapters.
        platform: L3 platform references (handler registry, config, budget).
        state: L3 ExecutionState for snapshot/time-travel.
        validation: Optional pre-flight validation pipeline.
        config: Orchestrator-wide defaults (timeout, retries).
    """

    def __init__(
        self,
        *,
        adapter_registry: AdapterRegistry,
        platform: PlatformRef,
        state: ExecutionStateProtocol,
        validation: ValidationPipeline | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._adapter_registry = adapter_registry
        self._platform = platform
        self._state = state
        self._validation = validation or ValidationPipeline()
        self._config = config or platform.config

        self._execution_dag = ExecutionDAG()
        self._interrupted = False
        self._status = DAGRunStatus.PENDING

    # ── Public API ──

    async def run(self, dag: PlanDAG, data: Any = None) -> ExecutionDAG:
        """Execute a PlanDAG sequentially and return the ExecutionDAG.

        Raises:
            InterruptError: If ``interrupt()`` was called during traversal.
        """
        self._execution_dag = ExecutionDAG()
        self._status = DAGRunStatus.RUNNING

        self._state.add_level()
        try:
            ready = dag.get_initial_nodes()
            completed: set[str] = set()

            while ready:
                for node_id in ready:
                    if self._interrupted:
                        self._status = DAGRunStatus.INTERRUPTED
                        raise InterruptError("DAG execution interrupted by user")

                    node = dag.get_node(node_id)

                    # Skip FlowNodes for now (Phase 5)
                    if isinstance(node, FlowNode):
                        self._execution_dag.mark_skipped(node)
                        completed.add(node_id)
                        continue

                    result = await self._run_node(node, data, completed, dag)
                    completed.add(node_id)

                    if not result.ok:
                        self._status = DAGRunStatus.FAILED
                        return self._execution_dag

                ready = dag.get_newly_ready_nodes(completed)

            self._status = DAGRunStatus.SUCCESS
            return self._execution_dag
        except InterruptError:
            self._status = DAGRunStatus.INTERRUPTED
            raise
        except Exception:
            self._status = DAGRunStatus.FAILED
            raise
        finally:
            self._state.remove_level()

    def interrupt(self) -> None:
        """Request a graceful interrupt of the current run."""
        self._interrupted = True

    @property
    def status(self) -> DAGRunStatus:
        """Current run status."""
        return self._status

    @property
    def execution_dag(self) -> ExecutionDAG:
        """The current execution DAG (in-progress or completed)."""
        return self._execution_dag

    # ── Internal ──

    async def _run_node(
        self,
        node: BaseNode,
        data: Any,
        completed: set[str],
        dag: PlanDAG,
    ) -> NodeResult:
        """Validate → execute → snapshot → record for a single node."""
        # 1. Record start
        self._execution_dag.record_start(node)

        # 2. Build validation context
        predecessors = dag.predecessors(node.id)
        ctx = ValidationContext(
            completed_node_ids=set(completed),
            ext={
                "predecessors": predecessors,
            },
        )
        if self._platform.budget_tracker is not None:
            ctx.ext["budget_tracker"] = self._platform.budget_tracker

        # 3. Validate
        validation_result = await self._validation.validate(node, data, ctx)
        if not validation_result.ok:
            result = NodeResult.failure(
                error_message=validation_result.error_message or "Validation failed",
                error_category="VALIDATION",
            )
            self._execution_dag.record(node, result)
            return result

        # 4. Execute via adapter
        try:
            result = await self._adapter_registry.execute(
                node, data, None, self._platform
            )
        except Exception as exc:
            result = NodeResult.failure(
                error_message=str(exc),
                error_category="EXECUTION",
            )

        # 5. Snapshot state
        version = self._state.snapshot()

        # 6. Record result
        self._execution_dag.record(node, result, state_version=version)

        return result
