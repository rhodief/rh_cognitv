"""
Execution Platform — Layer 3.

Self-contained, reusable runtime engine.
"""

from .errors import (
    BudgetError,
    CognitivError,
    ErrorCategory,
    EscalationError,
    InterruptError,
    LLMTransientError,
    PermanentError,
    TimeoutError,
    TransientError,
    ValidationError,
)
from .models import (
    Artifact,
    ArtifactProvenance,
    ArtifactStatus,
    ArtifactType,
    BaseEntry,
    BudgetSnapshot,
    EntryContent,
    EventKind,
    EventStatus,
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    Memory,
    MemoryOrigin,
    MemoryQuery,
    MemoryRole,
    MemoryShape,
    Provenance,
    QueryResult,
    ResultMetadata,
    TimeInfo,
    TokenBudget,
    TokenUsage,
    ToolResultData,
)
from .protocols import (
    BudgetTrackerProtocol,
    ContextStoreProtocol,
    EventBusProtocol,
    EventHandlerProtocol,
    ExecutionStateProtocol,
    HandlerRegistryProtocol,
    JsonSnapshotSerializer,
    LogCollectorProtocol,
    MiddlewareProtocol,
    PolicyChainProtocol,
    PolicyProtocol,
    SnapshotSerializerProtocol,
    TraceCollectorProtocol,
)
from .types import ID, Ext, Timestamp, generate_ulid, now_timestamp, parse_timestamp
from .types import EntryRef
from .budget import BudgetTracker
from .event_bus import EventBus
from .events import (
    DataPayload,
    EscalationRequested,
    EscalationResolved,
    ExecutionEvent,
    FunctionPayload,
    TextPayload,
    ToolPayload,
)
from .handlers import (
    DataHandler,
    FunctionHandler,
    HandlerRegistry,
    TextHandler,
    ToolHandler,
)
from .policies import (
    BudgetPolicy,
    PolicyChain,
    RetryPolicy,
    TimeoutPolicy,
)
from .state import ExecutionState
from .state_middleware import StateSnapshotMiddleware

__all__ = [
    # Types
    "ID",
    "Timestamp",
    "Ext",
    "generate_ulid",
    "now_timestamp",
    "parse_timestamp",
    # Enums
    "MemoryRole",
    "MemoryShape",
    "MemoryOrigin",
    "ArtifactType",
    "ArtifactStatus",
    "EventKind",
    "EventStatus",
    "ErrorCategory",
    # Models
    "EntryContent",
    "Provenance",
    "ArtifactProvenance",
    "TimeInfo",
    "BaseEntry",
    "Memory",
    "Artifact",
    "MemoryQuery",
    "QueryResult",
    "TokenBudget",
    "TokenUsage",
    "ResultMetadata",
    "LLMResultData",
    "FunctionResultData",
    "ToolResultData",
    "ExecutionResult",
    "BudgetSnapshot",
    # Errors
    "CognitivError",
    "TransientError",
    "PermanentError",
    "BudgetError",
    "InterruptError",
    "EscalationError",
    "LLMTransientError",
    "TimeoutError",
    "ValidationError",
    # Protocols
    "SnapshotSerializerProtocol",
    "JsonSnapshotSerializer",
    "EventBusProtocol",
    "MiddlewareProtocol",
    "EventHandlerProtocol",
    "HandlerRegistryProtocol",
    "ExecutionStateProtocol",
    "ContextStoreProtocol",
    "PolicyProtocol",
    "PolicyChainProtocol",
    "BudgetTrackerProtocol",
    "LogCollectorProtocol",
    "TraceCollectorProtocol",
    # Concrete implementations — Phase 2
    "EventBus",
    "BudgetTracker",
    # Phase 3 — Events
    "ExecutionEvent",
    "TextPayload",
    "DataPayload",
    "FunctionPayload",
    "ToolPayload",
    "EscalationRequested",
    "EscalationResolved",
    # Phase 3 — Handlers
    "HandlerRegistry",
    "TextHandler",
    "DataHandler",
    "FunctionHandler",
    "ToolHandler",
    # Phase 3 — Policies
    "PolicyChain",
    "RetryPolicy",
    "TimeoutPolicy",
    "BudgetPolicy",
    # Phase 3 — Types
    "EntryRef",
    # Phase 4 — State Management
    "ExecutionState",
    "StateSnapshotMiddleware",
]
