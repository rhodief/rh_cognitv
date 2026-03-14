#!/usr/bin/env python3
"""
Example: Simple Agent Orchestration with Terminal Progress
==========================================================

A user asks a question → a HelpfulAssistantSkill creates a multi-step plan:

  1. research    — (text)     gather background info
  2. web_search  — (function) mock web-search tool
  3. calculator  — (function) mock calculator tool
  4. draft       — (text)     compose an answer
  5. review      — (text)     review the answer quality

The EventBus prints every lifecycle event to the terminal so you can
watch the DAG execute in real time.

Run:
    python example/simple_agent.py
    python example/simple_agent.py "What is the speed of light in water?"
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from pydantic import BaseModel

# ── L1: Cognitive Layer ───────────────────────
from rh_cognitv.cognitive.adapters import (
    OrchestratorResult,
    ResultAdapter,
    SkillToDAGAdapter,
)
from rh_cognitv.cognitive.models import (
    FunctionStepConfig,
    SkillConstraints,
    SkillContext,
    SkillPlan,
    SkillProvenance,
    SkillResult,
    SkillStep,
    TextStepConfig,
)
from rh_cognitv.cognitive.skill import Skill

# ── L2: Orchestrator Layer ────────────────────
from rh_cognitv.orchestrator.adapters import AdapterRegistry, PlatformRef
from rh_cognitv.orchestrator.dag_orchestrator import DAGOrchestrator
from rh_cognitv.orchestrator.models import OrchestratorConfig

# ── L3: Execution Platform ────────────────────
from rh_cognitv.execution_platform.event_bus import EventBus
from rh_cognitv.execution_platform.events import ExecutionEvent
from rh_cognitv.execution_platform.handlers import HandlerRegistry
from rh_cognitv.execution_platform.models import (
    EventKind,
    ExecutionResult,
    FunctionResultData,
    LLMResultData,
    ResultMetadata,
    TokenUsage,
)
from rh_cognitv.execution_platform.protocols import EventHandlerProtocol
from rh_cognitv.execution_platform.state import ExecutionState


# ╔══════════════════════════════════════════════╗
# ║  Terminal Colors                             ║
# ╚══════════════════════════════════════════════╝

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RED = "\033[31m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_YELLOW = "\033[43m"


def header(text: str) -> None:
    print(f"\n{BOLD}{MAGENTA}{'═' * 60}{RESET}")
    print(f"{BOLD}{MAGENTA}  {text}{RESET}")
    print(f"{BOLD}{MAGENTA}{'═' * 60}{RESET}\n")


def step_log(icon: str, msg: str) -> None:
    print(f"  {icon}  {msg}")


# ╔══════════════════════════════════════════════╗
# ║  Mock LLM Handlers (simulate real LLM)      ║
# ╚══════════════════════════════════════════════╝


CANNED_RESPONSES: dict[str, str] = {
    "research": (
        "Background: Light travels at approximately 3×10⁸ m/s in vacuum. "
        "When entering a medium like water, it slows down proportionally "
        "to the refractive index of that medium (n ≈ 1.33 for water)."
    ),
    "draft": (
        "The speed of light in water is approximately 2.25 × 10⁸ m/s. "
        "This is because water has a refractive index of about 1.33, "
        "which means light travels 1.33 times slower in water than in "
        "a vacuum. The calculation is: c/n = 3×10⁸ / 1.33 ≈ 2.25×10⁸ m/s."
    ),
    "review": json.dumps({
        "passed": True,
        "feedback": "Answer is accurate, well-structured, and includes the calculation.",
        "issues": [],
    }),
}


class SimulatedTextHandler(EventHandlerProtocol[LLMResultData]):
    """Text handler that simulates LLM responses with a small delay."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[LLMResultData]:
        prompt = event.payload.prompt
        node_hint = event.ext.get("node_id", "")

        # Pick a canned response based on keywords in the prompt
        response = "I'll help with that."
        for key, text in CANNED_RESPONSES.items():
            if key in prompt.lower():
                response = text
                break

        # Simulate LLM thinking time
        await asyncio.sleep(0.3)

        tokens_in = len(prompt.split())
        tokens_out = len(response.split())

        return ExecutionResult(
            ok=True,
            value=LLMResultData(
                text=response,
                model="mock-gpt-4",
                token_usage=TokenUsage(
                    prompt_tokens=tokens_in,
                    completion_tokens=tokens_out,
                    total=tokens_in + tokens_out,
                ),
            ),
            metadata=ResultMetadata(),
        )


# ╔══════════════════════════════════════════════╗
# ║  Mock Tool Handlers (function nodes)         ║
# ╚══════════════════════════════════════════════╝


class MockFunctionHandler(EventHandlerProtocol[FunctionResultData]):
    """Function handler with built-in mock tools."""

    TOOLS = {
        "web_search": lambda args, kwargs: {
            "results": [
                {"title": "Speed of light - Wikipedia", "snippet": "In water, light speed ≈ 225,000 km/s"},
                {"title": "Refractive Index", "snippet": "Water n=1.33, glass n=1.5"},
            ],
            "query": kwargs.get("query", args[0] if args else ""),
        },
        "calculator": lambda args, kwargs: {
            "expression": kwargs.get("expression", args[0] if args else ""),
            "result": eval(kwargs.get("expression", args[0] if args else "0")),  # safe: controlled input
        },
    }

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any
    ) -> ExecutionResult[FunctionResultData]:
        fn_name = event.payload.function_name
        args = event.payload.args
        kwargs = event.payload.kwargs

        tool_fn = self.TOOLS.get(fn_name)
        if tool_fn is None:
            return ExecutionResult(
                ok=False,
                error_message=f"Unknown tool: {fn_name}",
                metadata=ResultMetadata(),
            )

        # Simulate tool execution time
        await asyncio.sleep(0.15)
        result = tool_fn(args, kwargs)

        return ExecutionResult(
            ok=True,
            value=FunctionResultData(
                return_value=result,
                duration_ms=150.0,
            ),
            metadata=ResultMetadata(),
        )


# ╔══════════════════════════════════════════════╗
# ║  EventBus Progress Logger                    ║
# ╚══════════════════════════════════════════════╝


class ProgressLogger:
    """Subscribes to the EventBus and prints DAG execution progress."""

    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._total_tokens: int = 0

    def elapsed(self) -> str:
        return f"{time.time() - self._start_time:.2f}s"

    def on_node_start(self, event: Any) -> None:
        node_id = event.get("node_id", "?")
        kind = event.get("kind", "?")
        icon = {"text": "🤖", "function": "🔧", "data": "📊"}.get(kind, "⚡")
        step_log(icon, f"{CYAN}START{RESET}  {BOLD}{node_id}{RESET}  ({kind})  {DIM}[{self.elapsed()}]{RESET}")

    def on_node_done(self, event: Any) -> None:
        node_id = event.get("node_id", "?")
        ok = event.get("ok", False)
        tokens = event.get("tokens", 0)
        self._total_tokens += tokens

        if ok:
            step_log("✅", f"{GREEN}DONE{RESET}   {BOLD}{node_id}{RESET}  {DIM}tokens={tokens}  [{self.elapsed()}]{RESET}")
        else:
            err = event.get("error", "unknown error")
            step_log("❌", f"{RED}FAIL{RESET}   {BOLD}{node_id}{RESET}  {err}  {DIM}[{self.elapsed()}]{RESET}")

    def on_node_result(self, event: Any) -> None:
        node_id = event.get("node_id", "?")
        preview = str(event.get("preview", ""))[:80]
        step_log("  ", f"{DIM}└─ {preview}{'…' if len(str(event.get('preview', ''))) > 80 else ''}{RESET}")

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# Simple event classes for the bus
class NodeStartEvent(BaseModel):
    node_id: str
    kind: str
    model_config = {"arbitrary_types_allowed": True}


class NodeDoneEvent(BaseModel):
    node_id: str
    ok: bool
    tokens: int = 0
    error: str = ""
    preview: str = ""
    model_config = {"arbitrary_types_allowed": True}


# ╔══════════════════════════════════════════════╗
# ║  HelpfulAssistantSkill                       ║
# ╚══════════════════════════════════════════════╝


class UserQuestion(BaseModel):
    """Input model for the assistant skill."""
    text: str


class HelpfulAssistantSkill(Skill):
    """Multi-step skill: research → tools → draft → review."""

    @property
    def name(self) -> str:
        return "helpful_assistant"

    @property
    def description(self) -> str:
        return "Researches a question, uses tools, drafts an answer, and reviews it."

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        question = input.text if hasattr(input, "text") else str(input)
        return SkillPlan(
            name="helpful_assistant",
            steps=[
                # Step 1: Research the topic
                SkillStep(
                    id="research",
                    kind="text",
                    config=TextStepConfig(
                        prompt=f"Research the following question and gather background information:\n\n{question}",
                        system_prompt="You are a research assistant. Provide factual background information.",
                        model="mock-gpt-4",
                    ),
                ),
                # Step 2: Web search (mock tool)
                SkillStep(
                    id="web_search",
                    kind="function",
                    config=FunctionStepConfig(
                        function_name="web_search",
                        kwargs={"query": question},
                    ),
                ),
                # Step 3: Calculator (mock tool) — runs in parallel with web_search
                SkillStep(
                    id="calculator",
                    kind="function",
                    config=FunctionStepConfig(
                        function_name="calculator",
                        kwargs={"expression": "3e8 / 1.33"},
                    ),
                    depends_on=["research"],  # only depends on research, parallel with web_search
                ),
                # Step 4: Draft answer (depends on all previous)
                SkillStep(
                    id="draft",
                    kind="text",
                    config=TextStepConfig(
                        prompt=(
                            f"Using the research and tool results, draft a clear, "
                            f"complete answer to: {question}"
                        ),
                        system_prompt=(
                            "You are a helpful assistant. Write a clear, accurate, "
                            "well-structured answer. Include relevant calculations."
                        ),
                        model="mock-gpt-4",
                    ),
                    depends_on=["web_search", "calculator"],
                ),
                # Step 5: Review the draft
                SkillStep(
                    id="review",
                    kind="text",
                    config=TextStepConfig(
                        prompt=(
                            "Review the following draft answer for accuracy and completeness. "
                            "Respond with JSON: {\"passed\": bool, \"feedback\": str, \"issues\": list}"
                        ),
                        system_prompt="You are a careful reviewer. Check for factual accuracy.",
                        model="mock-gpt-4",
                    ),
                ),
            ],
            constraints=SkillConstraints(timeout_seconds=30.0, max_retries=2),
        )

    async def interpret(self, result: Any) -> SkillResult:
        if not hasattr(result, "step_results"):
            return SkillResult(output=result, success=True,
                               provenance=SkillProvenance(skill_name=self.name))

        # Gather all step outputs
        steps = result.step_results
        draft = steps.get("draft")
        review = steps.get("review")

        if draft and draft.ok:
            answer = draft.value

            # Check review
            review_passed = True
            review_feedback = ""
            if review and review.ok and review.value:
                try:
                    review_data = json.loads(review.value)
                    review_passed = review_data.get("passed", True)
                    review_feedback = review_data.get("feedback", "")
                except (json.JSONDecodeError, TypeError):
                    review_feedback = str(review.value)

            return SkillResult(
                output={
                    "answer": answer,
                    "review_passed": review_passed,
                    "review_feedback": review_feedback,
                },
                success=True,
                provenance=SkillProvenance(skill_name=self.name),
            )

        error = draft.error_message if draft else "Draft step did not run"
        return SkillResult(
            output=None, success=False, error_message=error,
            provenance=SkillProvenance(skill_name=self.name),
        )


# ╔══════════════════════════════════════════════╗
# ║  Wire Everything Together                    ║
# ╚══════════════════════════════════════════════╝


async def run_agent(question: str) -> None:
    header("RH_COGNITV — Simple Agent Example")
    print(f"  {BOLD}Question:{RESET} {question}\n")

    # ── Set up EventBus + progress logger ─────
    bus = EventBus()
    logger = ProgressLogger()

    bus.on(NodeStartEvent, lambda e: logger.on_node_start({"node_id": e.node_id, "kind": e.kind}))
    bus.on(NodeDoneEvent, lambda e: logger.on_node_done({
        "node_id": e.node_id, "ok": e.ok, "tokens": e.tokens, "error": e.error, "preview": e.preview,
    }))
    bus.on(NodeDoneEvent, lambda e: logger.on_node_result({
        "node_id": e.node_id, "preview": e.preview,
    }))

    # ── L3: Handler Registry (with mock handlers) ─────
    handler_registry = HandlerRegistry()
    handler_registry.register(EventKind.TEXT, SimulatedTextHandler(bus))
    handler_registry.register(EventKind.FUNCTION, MockFunctionHandler(bus))

    # ── L2: Orchestrator wiring ─────
    config = OrchestratorConfig()
    platform = PlatformRef(registry=handler_registry, config=config)
    adapter_registry = AdapterRegistry.with_defaults()
    state = ExecutionState()

    # Wrap adapters to emit progress events on the bus
    original_execute = adapter_registry.execute

    async def tracked_execute(node, data, configs, plat):
        await bus.emit(NodeStartEvent(node_id=node.id, kind=node.kind))
        result = await original_execute(node, data, configs, plat)
        tokens = result.token_usage.total if result.token_usage else 0
        preview = str(result.value)[:120] if result.value else ""
        await bus.emit(NodeDoneEvent(
            node_id=node.id, ok=result.ok, tokens=tokens,
            error=result.error_message or "", preview=preview,
        ))
        return result

    adapter_registry.execute = tracked_execute  # type: ignore[assignment]

    orchestrator = DAGOrchestrator(
        adapter_registry=adapter_registry,
        platform=platform,
        state=state,
        config=config,
    )

    # ── L1: Skill → plan → adapt ─────
    skill = HelpfulAssistantSkill()
    input_data = UserQuestion(text=question)
    ctx = SkillContext()

    header("Phase 1: Planning")
    plan = await skill.plan(input_data, ctx)
    print(f"  Skill: {BOLD}{skill.name}{RESET}")
    print(f"  Steps: {len(plan.steps)}")
    for i, s in enumerate(plan.steps):
        deps = f" ← [{', '.join(s.depends_on)}]" if s.depends_on else ""
        icon = {"text": "🤖", "function": "🔧", "data": "📊"}.get(s.kind, "⚡")
        print(f"    {DIM}{i+1}.{RESET} {icon} {BOLD}{s.id}{RESET} ({s.kind}){deps}")

    # ── Adapt to PlanDAG ─────
    adapter = SkillToDAGAdapter()
    dag = adapter.to_dag(plan)
    print(f"\n  PlanDAG: {dag.node_count()} nodes, topology: {' → '.join(dag.topological_order())}")

    # ── Execute ─────
    header("Phase 2: Execution")
    exec_dag = await orchestrator.run(dag)

    # ── Adapt results back to L1 ─────
    header("Phase 3: Interpretation")
    orch_result = ResultAdapter().from_result(exec_dag)
    skill_result = await skill.interpret(orch_result)

    # ── Display results ─────
    header("Results")

    status_badge = f"{BG_GREEN}{BOLD} PASSED {RESET}" if skill_result.success else f"{BG_RED}{BOLD} FAILED {RESET}"
    print(f"  Status: {status_badge}")
    print(f"  Tokens: {logger.total_tokens}")
    print(f"  State snapshots: {state.version_count}")

    if skill_result.success and isinstance(skill_result.output, dict):
        out = skill_result.output
        print(f"\n  {BOLD}Answer:{RESET}")
        # Word-wrap the answer
        answer = out.get("answer", "")
        for line in _wrap(answer, 70):
            print(f"    {line}")

        review_badge = f"{GREEN}✓ Passed{RESET}" if out.get("review_passed") else f"{RED}✗ Failed{RESET}"
        print(f"\n  {BOLD}Review:{RESET} {review_badge}")
        print(f"    {DIM}{out.get('review_feedback', '')}{RESET}")
    elif skill_result.error_message:
        print(f"\n  {RED}Error: {skill_result.error_message}{RESET}")

    print(f"\n  {BOLD}Provenance:{RESET} {skill_result.provenance}")

    # ── Execution DAG dump ─────
    header("Execution DAG")
    for entry in exec_dag.entries():
        status_color = GREEN if entry.status.value == "success" else (RED if entry.status.value == "failed" else YELLOW)
        print(f"  {status_color}●{RESET} {entry.node_id:15s}  {status_color}{entry.status.value:12s}{RESET}  v{entry.state_version or '?'}")

    print(f"\n{DIM}  Done in {logger.elapsed()}{RESET}\n")


def _wrap(text: str, width: int) -> list[str]:
    """Simple word wrap."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return lines


# ╔══════════════════════════════════════════════╗
# ║  Main                                        ║
# ╚══════════════════════════════════════════════╝


if __name__ == "__main__":
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the speed of light in water?"
    )
    asyncio.run(run_agent(question))
