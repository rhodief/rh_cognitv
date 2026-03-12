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
    DAGTraversalState,
    FilterHandler,
    FilterNode,
    FlowHandlerRegistry,
    ForEachHandler,
    ForEachNode,
    GetHandler,
    GetNode,
    IfNotOkHandler,
    IfNotOkNode,
    Node,
    SwitchHandler,
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
from .plan_dag import (
    DAG,
    DAGBuilder,
    CycleError,
    DAGError,
    DisconnectedError,
    DuplicateEdgeError,
    DuplicateNodeError,
    FrozenDAGError,
    MissingNodeError,
    PlanDAG,
)
from .execution_dag import ExecutionDAG
from .adapters import (
    AdapterRegistry,
    DataNodeAdapter,
    FunctionNodeAdapter,
    PlatformRef,
    TextNodeAdapter,
    ToolNodeAdapter,
)
from .validation import (
    BudgetValidator,
    DependencyValidator,
    InputSchemaValidator,
    ValidationPipeline,
)
from .dag_orchestrator import DAGOrchestrator

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
    # Flow Handlers
    "FlowHandlerRegistry",
    "DAGTraversalState",
    "ForEachHandler",
    "FilterHandler",
    "SwitchHandler",
    "GetHandler",
    "IfNotOkHandler",
    # DAG Data Structures
    "DAG",
    "PlanDAG",
    "DAGBuilder",
    "ExecutionDAG",
    # DAG Errors
    "DAGError",
    "CycleError",
    "DisconnectedError",
    "DuplicateNodeError",
    "MissingNodeError",
    "DuplicateEdgeError",
    "FrozenDAGError",
    # Adapters
    "AdapterRegistry",
    "PlatformRef",
    "TextNodeAdapter",
    "DataNodeAdapter",
    "FunctionNodeAdapter",
    "ToolNodeAdapter",
    # Validation
    "ValidationPipeline",
    "InputSchemaValidator",
    "DependencyValidator",
    "BudgetValidator",
    # Orchestrator
    "DAGOrchestrator",
]
