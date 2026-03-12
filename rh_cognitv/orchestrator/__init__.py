"""
Orchestrator — Layer 2.

Strategy brain: translates high-level intent into Plan DAGs,
executes them via the Execution Platform, and records Execution DAGs.
"""

from .models import (
    DAGRunStatus,
    ExecutionDAGEntry,
    FlowResult,
    NodeExecutionStatus,
    NodeResult,
    OrchestratorConfig,
    ValidationContext,
    ValidationResult,
)
from .nodes import (
    BaseNode,
    DataNode,
    ExecutionNode,
    FlowNode,
    FunctionNode,
    TextNode,
    ToolNode,
)
from .flow_nodes import (
    CompositeNode,
    FilterNode,
    ForEachNode,
    GetNode,
    IfNotOkNode,
    Node,
    SwitchNode,
)
from .protocols import (
    DAGProtocol,
    FlowHandlerProtocol,
    NodeAdapterProtocol,
    NodeProtocol,
    NodeValidatorProtocol,
    OrchestratorProtocol,
    ValidationPipelineProtocol,
)

__all__ = [
    # Protocols
    "OrchestratorProtocol",
    "DAGProtocol",
    "NodeProtocol",
    "NodeAdapterProtocol",
    "FlowHandlerProtocol",
    "NodeValidatorProtocol",
    "ValidationPipelineProtocol",
    # Models
    "NodeResult",
    "ValidationResult",
    "ValidationContext",
    "FlowResult",
    "OrchestratorConfig",
    "NodeExecutionStatus",
    "ExecutionDAGEntry",
    "DAGRunStatus",
    # Nodes — Execution
    "BaseNode",
    "ExecutionNode",
    "FlowNode",
    "Node",
    "TextNode",
    "DataNode",
    "FunctionNode",
    "ToolNode",
    # Nodes — Flow
    "ForEachNode",
    "FilterNode",
    "SwitchNode",
    "GetNode",
    "IfNotOkNode",
    "CompositeNode",
]
