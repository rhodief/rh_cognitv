"""
DAGOrchestrator — core traversal engine.

DD-L2-03: Topological Sort + Ready Queue.
DI-L2-06: Interrupt check per-node.

The orchestrator:
1. Takes a frozen PlanDAG and data
2. Iterates through nodes in topological order via a ready queue
3. For each node: validate → execute via adapter → snapshot state → record
4. Parallel branches execute concurrently via asyncio.gather (Phase 6)
5. Returns the completed ExecutionDAG
"""

from __future__ import annotations

import asyncio
from typing import Any

from rh_cognitv.execution_platform.errors import InterruptError
from rh_cognitv.execution_platform.models import Artifact, Memory
from rh_cognitv.execution_platform.protocols import (
    ContextStoreProtocol,
    ExecutionStateProtocol,
)

from .adapters import AdapterRegistry, PlatformRef
from .execution_dag import ExecutionDAG
from .flow_nodes import DAGTraversalState, FlowHandlerRegistry
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
        flow_handler_registry: Optional FlowNode handler registry.
        context_store: Optional L3 ContextStore for resolving context_refs.
        context_serializer: Optional serializer for rendering resolved context.
    """

    def __init__(
        self,
        *,
        adapter_registry: AdapterRegistry,
        platform: PlatformRef,
        state: ExecutionStateProtocol,
        validation: ValidationPipeline | None = None,
        config: OrchestratorConfig | None = None,
        flow_handler_registry: FlowHandlerRegistry | None = None,
        context_store: ContextStoreProtocol | None = None,
        context_serializer: Any | None = None,
    ) -> None:
        self._adapter_registry = adapter_registry
        self._platform = platform
        self._state = state
        self._validation = validation or ValidationPipeline()
        self._config = config or platform.config
        self._flow_registry = flow_handler_registry or FlowHandlerRegistry.with_defaults()
        self._context_store = context_store
        self._context_serializer = context_serializer

        self._execution_dag = ExecutionDAG()
        self._interrupted = False
        self._status = DAGRunStatus.PENDING
        self._node_results: dict[str, NodeResult] = {}

    # ── Public API ──

    async def run(self, dag: PlanDAG, data: Any = None) -> ExecutionDAG:
        """Execute a PlanDAG and return the ExecutionDAG.

        Ready-queue nodes are executed in parallel via ``asyncio.gather``.
        If any node in a parallel batch fails, the DAG stops after the
        batch completes (remaining batches are not started).

        Raises:
            InterruptError: If ``interrupt()`` was called during traversal.
        """
        self._execution_dag = ExecutionDAG()
        self._node_results = {}
        self._status = DAGRunStatus.RUNNING
        current_data = data

        self._state.add_level()
        try:
            ready = dag.get_initial_nodes()
            completed: set[str] = set()

            while ready:
                if self._interrupted:
                    self._status = DAGRunStatus.INTERRUPTED
                    raise InterruptError("DAG execution interrupted by user")

                # Execute all ready nodes in parallel
                results = await asyncio.gather(
                    *[
                        self._process_node(node_id, current_data, completed, dag)
                        for node_id in ready
                    ],
                    return_exceptions=True,
                )

                # Process results from the parallel batch
                batch_failed = False
                for node_id, result in zip(ready, results):
                    if isinstance(result, InterruptError):
                        self._status = DAGRunStatus.INTERRUPTED
                        raise result
                    if isinstance(result, BaseException):
                        self._status = DAGRunStatus.FAILED
                        raise result
                    # result is (ok: bool)
                    completed.add(node_id)
                    if not result:
                        batch_failed = True

                if batch_failed:
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

    async def _process_node(
        self,
        node_id: str,
        data: Any,
        completed: set[str],
        dag: PlanDAG,
    ) -> bool:
        """Dispatch a single node (execution or flow). Returns True if ok."""
        if self._interrupted:
            raise InterruptError("DAG execution interrupted by user")

        node = dag.get_node(node_id)

        if isinstance(node, FlowNode):
            return await self._run_flow_node(node, data, completed, dag)

        result = await self._run_node(node, data, completed, dag)
        self._node_results[node_id] = result
        return result.ok

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

        # 3.5 Resolve context_refs (Phase 3.7)
        resolved_data = await self._resolve_context_refs(node, data)

        # 4. Execute via adapter
        try:
            result = await self._adapter_registry.execute(
                node, resolved_data, None, self._platform
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

    async def _resolve_context_refs(
        self, node: BaseNode, data: Any
    ) -> Any:
        """Resolve context_refs from node.ext and merge into data.

        If no context_store is configured or the node has no context_refs,
        returns data unchanged (backward compatible).
        """
        context_refs = node.ext.get("context_refs")
        if not context_refs or self._context_store is None:
            return data

        resolved: dict[str, Any] = dict(data) if isinstance(data, dict) else {}

        for ref in context_refs:
            kind = ref.get("kind")
            key = ref.get("key", "context")
            value = None

            if kind == "memory":
                ref_id = ref.get("id")
                if ref_id:
                    entry = await self._context_store.get(ref_id)
                    if entry is not None:
                        value = self._extract_text(entry)

            elif kind == "artifact":
                slug = ref.get("slug")
                version = ref.get("version")
                if slug:
                    entry = await self._context_store.get_artifact(slug, version)
                    if entry is not None:
                        value = self._extract_text(entry)

            elif kind == "query":
                from rh_cognitv.execution_platform.models import MemoryQuery

                query_data = ref.get("query")
                if query_data:
                    query = MemoryQuery.model_validate(query_data)
                    results = await self._context_store.recall(query)
                    entries = [r.entry for r in results]
                    if self._context_serializer is not None:
                        memories = [e for e in entries if isinstance(e, Memory)]
                        artifacts = [e for e in entries if isinstance(e, Artifact)]
                        value = self._context_serializer.serialize(memories, artifacts)
                    else:
                        value = "\n".join(
                            self._extract_text(e) for e in entries if e is not None
                        )

            elif kind == "previous_result":
                from_step = ref.get("from_step")
                if from_step and from_step in self._node_results:
                    value = self._node_results[from_step].value

            if value is not None:
                resolved[key] = value

        return resolved

    @staticmethod
    def _extract_text(entry: Any) -> str:
        """Extract display text from a Memory or Artifact."""
        if hasattr(entry, "content") and hasattr(entry.content, "text"):
            return entry.content.text
        return str(entry)

    async def _run_flow_node(
        self,
        node: FlowNode,
        data: Any,
        completed: set[str],
        dag: PlanDAG,
    ) -> bool:
        """Dispatch a FlowNode to its handler and process the FlowResult.

        Returns True if the flow succeeded, False if the DAG should stop.
        """
        from .flow_nodes import ForEachNode

        self._execution_dag.record_start(node)

        dag_state = DAGTraversalState(
            completed_node_ids=set(completed),
            execution_dag=self._execution_dag,
            node_results=dict(self._node_results),
        )

        try:
            flow_result = await self._flow_registry.handle(node, data, dag_state)
        except Exception as exc:
            result = NodeResult.failure(
                error_message=str(exc), error_category="FLOW"
            )
            self._execution_dag.record(node, result)
            return False

        if not flow_result.ok:
            result = NodeResult.failure(
                error_message=flow_result.error_message or "Flow handler failed",
                error_category="FLOW",
            )
            self._execution_dag.record(node, result)
            self._node_results[node.id] = result
            return False

        # Handle ForEach expansion: execute inner node for each item
        if flow_result.expanded_node_ids and isinstance(node, ForEachNode):
            inner_id = flow_result.expanded_node_ids[0]
            items = flow_result.data if flow_result.data is not None else []
            inner_node = dag.get_node(inner_id)

            if node.failure_strategy == "collect_all":
                # Run all items in parallel, collect partial results
                if items:
                    gather_results = await asyncio.gather(
                        *[
                            self._run_node(inner_node, item, completed, dag)
                            for item in items
                        ],
                        return_exceptions=True,
                    )
                    any_failed = False
                    for r in gather_results:
                        if isinstance(r, BaseException):
                            any_failed = True
                        elif isinstance(r, NodeResult):
                            self._node_results[inner_id] = r
                            if not r.ok:
                                any_failed = True
                    if any_failed:
                        fail_result = NodeResult.failure(
                            error_message=f"ForEach collect_all: some iterations of '{inner_id}' failed",
                            error_category="FLOW",
                        )
                        self._execution_dag.record(node, fail_result)
                        self._node_results[node.id] = fail_result
                        return False
            else:
                # fail_fast: stop on first failure (sequential)
                for item in items:
                    inner_result = await self._run_node(
                        inner_node, item, completed, dag
                    )
                    self._node_results[inner_id] = inner_result
                    if not inner_result.ok:
                        fail_result = NodeResult.failure(
                            error_message=f"ForEach inner node '{inner_id}' failed",
                            error_category="FLOW",
                        )
                        self._execution_dag.record(node, fail_result)
                        self._node_results[node.id] = fail_result
                        return False

            completed.add(inner_id)

        # Handle skipped nodes
        for skip_id in flow_result.skipped_node_ids:
            try:
                skip_node = dag.get_node(skip_id)
                self._execution_dag.mark_skipped(skip_node)
                completed.add(skip_id)
            except KeyError:
                pass

        # Handle redirect
        if flow_result.redirect_to:
            try:
                redirect_node = dag.get_node(flow_result.redirect_to)
                redirect_result = await self._run_node(
                    redirect_node, flow_result.data or data, completed, dag
                )
                self._node_results[flow_result.redirect_to] = redirect_result
                completed.add(flow_result.redirect_to)
            except KeyError:
                fail_result = NodeResult.failure(
                    error_message=f"Redirect target '{flow_result.redirect_to}' not found",
                    error_category="FLOW",
                )
                self._execution_dag.record(node, fail_result)
                self._node_results[node.id] = fail_result
                return False

        # Record success for the flow node itself
        flow_node_result = NodeResult.success(value=flow_result.data)
        self._execution_dag.record(node, flow_node_result)
        self._node_results[node.id] = flow_node_result
        return True
