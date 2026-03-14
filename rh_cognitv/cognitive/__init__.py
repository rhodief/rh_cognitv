"""
Cognitive Layer — Layer 1.

Intelligence layer: skills, LLM abstraction, prompt building,
context serialization, and the L1→L2 adapter.
"""

from .models import (
    BuiltPrompt,
    CompletionResult,
    ContextRef,
    CreateArtifact,
    CreateMemory,
    DataStepConfig,
    FunctionStepConfig,
    Message,
    MessageRole,
    ReplanRequest,
    SkillConfig,
    SkillConstraints,
    SkillContext,
    SkillPlan,
    SkillProvenance,
    SkillResult,
    SkillStep,
    TextStepConfig,
    ToolCall,
    ToolResult,
    ToolStepConfig,
)
from .protocols import (
    ContextSerializerProtocol,
    EmbeddingProtocol,
    LLMProtocol,
    MetaSkillProtocol,
    PromptProtocol,
    SkillProtocol,
    StreamingLLMProtocol,
)
from .llm import (
    CallLog,
    CompletionCall,
    MockLLM,
    StructuredCall,
    ToolsCall,
)
from .skill import (
    ConfigSkill,
    RetryableValidationError,
    Skill,
)
from .prompt import (
    PromptBuilder,
    TemplateRenderer,
)
from .serializer import (
    NaiveSerializer,
    SectionSerializer,
)
from .adapters import (
    OrchestratorResult,
    ResultAdapter,
    SkillToDAGAdapter,
)
from .builtin_skills import (
    CodeGenerationSkill,
    DataExtractionSkill,
    ReviewSkill,
    TextGenerationSkill,
)
from .meta_skill import MetaSkill

__all__ = [
    # Protocols
    "SkillProtocol",
    "LLMProtocol",
    "StreamingLLMProtocol",
    "EmbeddingProtocol",
    "PromptProtocol",
    "ContextSerializerProtocol",
    "MetaSkillProtocol",
    # Models — Messages & LLM
    "MessageRole",
    "Message",
    "CompletionResult",
    "ToolCall",
    "ToolResult",
    "BuiltPrompt",
    # Models — Context Ref
    "ContextRef",
    # Models — Step Config
    "TextStepConfig",
    "DataStepConfig",
    "FunctionStepConfig",
    "ToolStepConfig",
    # Models — Plan
    "SkillStep",
    "SkillConstraints",
    "SkillPlan",
    # Models — Context
    "SkillContext",
    # Models — Result
    "SkillProvenance",
    "CreateMemory",
    "CreateArtifact",
    "ReplanRequest",
    "SkillResult",
    # Models — Config
    "SkillConfig",
    # LLM
    "MockLLM",
    "CallLog",
    "CompletionCall",
    "StructuredCall",
    "ToolsCall",
    # Skill
    "Skill",
    "ConfigSkill",
    "RetryableValidationError",
    # Prompt
    "PromptBuilder",
    "TemplateRenderer",
    # Serializer
    "NaiveSerializer",
    "SectionSerializer",
    # Adapters
    "OrchestratorResult",
    "SkillToDAGAdapter",
    "ResultAdapter",
    # Built-in Skills
    "TextGenerationSkill",
    "DataExtractionSkill",
    "CodeGenerationSkill",
    "ReviewSkill",
    # MetaSkill
    "MetaSkill",
]
