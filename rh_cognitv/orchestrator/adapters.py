"""
Adapter Registry — L2 → L3 bridge.

Each concrete adapter converts an L2 node into an L3 ExecutionEvent,
builds a per-node PolicyChain (with merged timeout/retry from node
overrides + orchestrator defaults), executes via PolicyChain, and
normalises the ExecutionResult into a uniform NodeResult.

DD-L2-04: Registry (Strategy per Node Kind).
DI-L2-07: Per-node timeout/retry config with orchestrator defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rh_cognitv.execution_platform.events import (
    DataPayload,
    ExecutionEvent,
    FunctionPayload,
    TextPayload,
    ToolPayload,
)
from rh_cognitv.execution_platform.models import EventKind
from rh_cognitv.execution_platform.policies import (
    BudgetPolicy,
    PolicyChain,
    RetryPolicy,
    TimeoutPolicy,
)
from rh_cognitv.execution_platform.protocols import (
    BudgetTrackerProtocol,
    HandlerRegistryProtocol,
)

from .models import NodeResult, OrchestratorConfig
from .nodes import BaseNode, DataNode, FunctionNode, TextNode, ToolNode
from .protocols import NodeAdapterProtocol


# ──────────────────────────────────────────────
# PlatformRef — L3 references for adapters
# ──────────────────────────────────────────────


@dataclass
class PlatformRef:
    """Bundle of L3 references that adapters need for execution.

    Adapters receive this instead of depending on the full orchestrator.
    """

    registry: HandlerRegistryProtocol
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    budget_tracker: BudgetTrackerProtocol | None = None

    def build_policy_chain(self, node: BaseNode) -> PolicyChain:
        """Build a per-node PolicyChain with merged timeout/retry overrides."""
        timeout = node.timeout_seconds or self.config.default_timeout_seconds
        retries = node.max_retries or self.config.default_max_retries
        base_delay = self.config.default_retry_base_delay

        policies: list[Any] = []
        if self.budget_tracker is not None:
            policies.append(BudgetPolicy(tracker=self.budget_tracker))
        policies.append(TimeoutPolicy(seconds=timeout))
        policies.append(RetryPolicy(max_attempts=retries, base_delay=base_delay))
        return PolicyChain(policies)


# ──────────────────────────────────────────────
# Adapter Registry
# ──────────────────────────────────────────────


class AdapterRegistry:
    """Maps node ``kind`` values to concrete adapters.

    Use ``with_defaults()`` to get a registry pre-loaded with the four
    built-in execution-node adapters.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, NodeAdapterProtocol] = {}

    def register(self, kind: str, adapter: NodeAdapterProtocol) -> None:
        """Register an adapter for a node kind."""
        self._adapters[kind] = adapter

    def get(self, kind: str) -> NodeAdapterProtocol | None:
        """Retrieve the adapter for a kind, or ``None``."""
        return self._adapters.get(kind)

    async def execute(
        self,
        node: BaseNode,
        data: Any,
        configs: Any,
        platform: PlatformRef,
    ) -> NodeResult:
        """Dispatch to the adapter registered for ``node.kind``."""
        adapter = self._adapters.get(node.kind)
        if adapter is None:
            return NodeResult.failure(
                error_message=f"No adapter registered for kind '{node.kind}'",
            )
        return await adapter.execute(node, data, configs, platform)

    @property
    def registered_kinds(self) -> list[str]:
        """Return the list of registered node kinds."""
        return list(self._adapters)

    @classmethod
    def with_defaults(cls) -> AdapterRegistry:
        """Create a registry pre-loaded with the four built-in adapters."""
        registry = cls()
        registry.register("text", TextNodeAdapter())
        registry.register("data", DataNodeAdapter())
        registry.register("function", FunctionNodeAdapter())
        registry.register("tool", ToolNodeAdapter())
        return registry


# ──────────────────────────────────────────────
# Concrete Adapters
# ──────────────────────────────────────────────


class TextNodeAdapter(NodeAdapterProtocol):
    """Adapter for TextNode → EventKind.TEXT."""

    async def execute(
        self,
        node: BaseNode,
        data: Any,
        configs: Any,
        platform: Any,
    ) -> NodeResult:
        assert isinstance(node, TextNode)
        event = ExecutionEvent(
            kind=EventKind.TEXT,
            payload=TextPayload(
                prompt=node.prompt,
                system_prompt=node.system_prompt,
                model=node.model,
                temperature=node.temperature,
                max_tokens=node.max_tokens,
            ),
        )
        chain = platform.build_policy_chain(node)
        result = await chain(platform.registry.handle, event, data, configs)
        return NodeResult.from_execution_result(result)


class DataNodeAdapter(NodeAdapterProtocol):
    """Adapter for DataNode → EventKind.DATA."""

    async def execute(
        self,
        node: BaseNode,
        data: Any,
        configs: Any,
        platform: Any,
    ) -> NodeResult:
        assert isinstance(node, DataNode)
        event = ExecutionEvent(
            kind=EventKind.DATA,
            payload=DataPayload(
                prompt=node.prompt,
                output_schema=node.output_schema,
                model=node.model,
            ),
        )
        chain = platform.build_policy_chain(node)
        result = await chain(platform.registry.handle, event, data, configs)
        return NodeResult.from_execution_result(result)


class FunctionNodeAdapter(NodeAdapterProtocol):
    """Adapter for FunctionNode → EventKind.FUNCTION."""

    async def execute(
        self,
        node: BaseNode,
        data: Any,
        configs: Any,
        platform: Any,
    ) -> NodeResult:
        assert isinstance(node, FunctionNode)
        event = ExecutionEvent(
            kind=EventKind.FUNCTION,
            payload=FunctionPayload(
                function_name=node.function_name,
                args=node.args,
                kwargs=node.kwargs,
            ),
        )
        chain = platform.build_policy_chain(node)
        result = await chain(platform.registry.handle, event, data, configs)
        return NodeResult.from_execution_result(result)


class ToolNodeAdapter(NodeAdapterProtocol):
    """Adapter for ToolNode → EventKind.TOOL."""

    async def execute(
        self,
        node: BaseNode,
        data: Any,
        configs: Any,
        platform: Any,
    ) -> NodeResult:
        assert isinstance(node, ToolNode)
        event = ExecutionEvent(
            kind=EventKind.TOOL,
            payload=ToolPayload(
                prompt=node.prompt,
                tools=node.tools,
                model=node.model,
            ),
        )
        chain = platform.build_policy_chain(node)
        result = await chain(platform.registry.handle, event, data, configs)
        return NodeResult.from_execution_result(result)
