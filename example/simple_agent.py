#!/usr/bin/env python3
"""
Simple Agent Example — full L1 → L2 → L3 pipeline with OpenAI.
================================================================

A user asks a question → HelpfulAssistantSkill creates a multi-step plan:

  1. research_france  — (function)  mock lookup tool
  2. research_eu      — (function)  mock lookup tool
  3. calculate_pct    — (function)  mock calculator tool
  4. draft            — (text)      OpenAI drafts the answer
  5. review           — (text)      OpenAI reviews and polishes

The EventBus prints every step start/complete to the terminal so you
can watch the DAG execute in real time.

Run::

    python example/simple_agent.py
    python example/simple_agent.py "What is the GDP of Germany?"
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv
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
from rh_cognitv.cognitive.protocols import SkillProtocol

# ── L2: Orchestrator Layer ────────────────────
from rh_cognitv.orchestrator.adapters import AdapterRegistry, PlatformRef
from rh_cognitv.orchestrator.dag_orchestrator import DAGOrchestrator
from rh_cognitv.orchestrator.models import OrchestratorConfig

# ── L3: Execution Platform ────────────────────
from rh_cognitv.execution_platform.event_bus import EventBus
from rh_cognitv.execution_platform.events import (
    ExecutionEvent,
    FunctionPayload,
    TextPayload,
)
from rh_cognitv.execution_platform.handlers import HandlerRegistry
from rh_cognitv.execution_platform.models import (
    EventKind,
    ExecutionResult,
    FunctionResultData,
    ResultMetadata,
)
from rh_cognitv.execution_platform.openai_handler import OpenAITextHandler
from rh_cognitv.execution_platform.protocols import EventHandlerProtocol
from rh_cognitv.execution_platform.state import ExecutionState


# ╔══════════════════════════════════════════════╗
# ║  Terminal Formatting                         ║
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


def header(text: str) -> None:
    print(f"\n{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"{BOLD}{MAGENTA}  {text}{RESET}")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}\n")


# ╔══════════════════════════════════════════════╗
# ║  EventBus Custom Events                      ║
# ╚══════════════════════════════════════════════╝


@dataclass
class StepStarted:
    step_id: str
    kind: str
    description: str


@dataclass
class StepCompleted:
    step_id: str
    kind: str
    success: bool
    preview: str
    duration_ms: float


@dataclass
class AgentStarted:
    question: str


@dataclass
class AgentCompleted:
    success: bool
    answer: str


def setup_event_bus() -> EventBus:
    """Create an EventBus with coloured terminal logging subscribers."""
    bus = EventBus()

    def on_agent_started(event: AgentStarted) -> None:
        header("Agent Started")
        print(f"  {BOLD}Question:{RESET} {event.question}\n")

    def on_step_started(event: StepStarted) -> None:
        kind_color = {"function": YELLOW, "text": MAGENTA}.get(event.kind, BLUE)
        icon = ">" if event.kind == "function" else "*"
        print(
            f"  {kind_color}[{icon}] {event.step_id}{RESET}  "
            f"{event.description}"
        )

    def on_step_completed(event: StepCompleted) -> None:
        tag = f"{GREEN}OK{RESET}" if event.success else f"{RED}FAIL{RESET}"
        print(
            f"      {tag}  {event.duration_ms:.0f}ms  "
            f"{DIM}{event.preview}{RESET}\n"
        )

    def on_agent_completed(event: AgentCompleted) -> None:
        border = "=" * 60
        if event.success:
            print(f"\n{BOLD}{GREEN}{border}{RESET}")
            print(f"{BOLD}{GREEN}  Agent Completed Successfully{RESET}")
            print(f"{BOLD}{GREEN}{border}{RESET}\n")
        else:
            print(f"\n{BOLD}{RED}{border}{RESET}")
            print(f"{BOLD}{RED}  Agent Failed{RESET}")
            print(f"{BOLD}{RED}{border}{RESET}\n")

    bus.on(AgentStarted, on_agent_started)
    bus.on(StepStarted, on_step_started)
    bus.on(StepCompleted, on_step_completed)
    bus.on(AgentCompleted, on_agent_completed)

    return bus


# ╔══════════════════════════════════════════════╗
# ║  Mock Tools                                  ║
# ╚══════════════════════════════════════════════╝


def lookup(topic: str) -> dict[str, Any]:
    """Simulated knowledge-base lookup."""
    db: dict[str, dict[str, Any]] = {
        "france_population": {
            "country": "France",
            "population": 68_042_591,
            "year": 2024,
            "source": "INSEE estimate",
        },
        "eu_population": {
            "entity": "European Union",
            "population": 448_400_000,
            "year": 2024,
            "member_states": 27,
            "source": "Eurostat estimate",
        },
    }
    return db.get(topic, {"error": f"No data found for '{topic}'"})


def calculate(expression: str) -> dict[str, Any]:
    """Simulated safe calculator (pre-computed results only)."""
    known: dict[str, dict[str, Any]] = {
        "68042591 / 448400000 * 100": {
            "expression": "68042591 / 448400000 * 100",
            "result": 15.18,
            "formatted": "15.18%",
        },
    }
    return known.get(expression, {"error": f"Unknown expression: {expression}"})


# ╔══════════════════════════════════════════════╗
# ║  Custom FunctionHandler (dispatches tools)   ║
# ╚══════════════════════════════════════════════╝


class ToolkitFunctionHandler(EventHandlerProtocol[FunctionResultData]):
    """FunctionHandler that dispatches to registered callables.

    Emits StepStarted / StepCompleted to the EventBus for terminal tracking.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._functions: dict[str, Callable[..., Any]] = {}
        self._bus = event_bus

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._functions[name] = fn

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any,
    ) -> ExecutionResult[FunctionResultData]:
        payload = event.payload
        if not isinstance(payload, FunctionPayload):
            return ExecutionResult(
                ok=False,
                error_message=f"Expected FunctionPayload, got {type(payload).__name__}",
            )

        fn = self._functions.get(payload.function_name)
        if fn is None:
            return ExecutionResult(
                ok=False,
                error_message=f"Unknown function: {payload.function_name}",
            )

        # ── Emit start ──
        args_preview = ", ".join(
            f"{k}={v!r}" for k, v in payload.kwargs.items()
        ) or ", ".join(repr(a) for a in payload.args)
        if self._bus:
            await self._bus.emit(
                StepStarted(
                    step_id=payload.function_name,
                    kind="function",
                    description=f"{payload.function_name}({args_preview})",
                )
            )

        # ── Execute ──
        start = time.perf_counter()
        if asyncio.iscoroutinefunction(fn):
            result = await fn(*payload.args, **payload.kwargs)
        else:
            result = fn(*payload.args, **payload.kwargs)
        duration_ms = (time.perf_counter() - start) * 1000

        # ── Emit completion ──
        if self._bus:
            await self._bus.emit(
                StepCompleted(
                    step_id=payload.function_name,
                    kind="function",
                    success=True,
                    preview=str(result)[:100],
                    duration_ms=duration_ms,
                )
            )

        return ExecutionResult(
            ok=True,
            value=FunctionResultData(return_value=result, duration_ms=duration_ms),
            metadata=ResultMetadata(duration_ms=duration_ms),
        )


# ╔══════════════════════════════════════════════╗
# ║  Tracked OpenAI Handler (+ EventBus)         ║
# ╚══════════════════════════════════════════════╝


class TrackedOpenAITextHandler(OpenAITextHandler):
    """OpenAI TextHandler that emits EventBus events for terminal tracking."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        event_bus: EventBus | None = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model)
        self._bus = event_bus

    async def __call__(
        self, event: ExecutionEvent, data: Any, configs: Any,
    ) -> ExecutionResult[Any]:
        payload = event.payload

        # ── Emit start ──
        label = "llm"
        if isinstance(payload, TextPayload):
            label = payload.model or self._model
            prompt_preview = payload.prompt[:60].replace("\n", " ")
            if self._bus:
                await self._bus.emit(
                    StepStarted(
                        step_id=label,
                        kind="text",
                        description=f"OpenAI {label}: {prompt_preview}...",
                    )
                )

        # ── Execute ──
        start = time.perf_counter()
        result = await super().__call__(event, data, configs)
        duration_ms = (time.perf_counter() - start) * 1000

        # ── Emit completion ──
        if self._bus:
            preview = ""
            if result.ok and result.value:
                preview = result.value.text[:100].replace("\n", " ")
            await self._bus.emit(
                StepCompleted(
                    step_id=label,
                    kind="text",
                    success=result.ok,
                    preview=preview,
                    duration_ms=duration_ms,
                )
            )

        return result


# ╔══════════════════════════════════════════════╗
# ║  User Input                                  ║
# ╚══════════════════════════════════════════════╝


class UserQuestion(BaseModel):
    text: str


# ╔══════════════════════════════════════════════╗
# ║  HelpfulAssistantSkill                       ║
# ╚══════════════════════════════════════════════╝


class HelpfulAssistantSkill(SkillProtocol):
    """Multi-step skill: research -> calculate -> draft -> review."""

    @property
    def name(self) -> str:
        return "helpful_assistant"

    @property
    def description(self) -> str:
        return "Answers user questions using research tools and LLM analysis"

    async def plan(self, input: BaseModel, context: SkillContext) -> SkillPlan:
        question = input.text if hasattr(input, "text") else str(input)

        return SkillPlan(
            name="helpful_assistant",
            steps=[
                # Step 1: Research France population
                SkillStep(
                    id="research_france",
                    kind="function",
                    config=FunctionStepConfig(
                        function_name="lookup",
                        kwargs={"topic": "france_population"},
                    ),
                ),
                # Step 2: Research EU population
                SkillStep(
                    id="research_eu",
                    kind="function",
                    config=FunctionStepConfig(
                        function_name="lookup",
                        kwargs={"topic": "eu_population"},
                    ),
                ),
                # Step 3: Calculate percentage
                SkillStep(
                    id="calculate_pct",
                    kind="function",
                    config=FunctionStepConfig(
                        function_name="calculate",
                        kwargs={"expression": "68042591 / 448400000 * 100"},
                    ),
                ),
                # Step 4: Draft answer with OpenAI
                SkillStep(
                    id="draft",
                    kind="text",
                    config=TextStepConfig(
                        prompt=(
                            f"User question: {question}\n\n"
                            "Research data gathered:\n"
                            "- France population (2024): 68,042,591 "
                            "(source: INSEE)\n"
                            "- EU population (2024): 448,400,000 across "
                            "27 member states (source: Eurostat)\n"
                            "- France as percentage of EU: 15.18%\n\n"
                            "Write a clear, well-structured answer "
                            "addressing the user's question. "
                            "Include the key figures with sources."
                        ),
                        system_prompt=(
                            "You are a knowledgeable research assistant. "
                            "Provide accurate, well-cited answers based "
                            "on the data provided."
                        ),
                        model="gpt-4o-mini",
                        temperature=0.7,
                        max_tokens=500,
                    ),
                ),
                # Step 5: Review and polish with OpenAI
                SkillStep(
                    id="review",
                    kind="text",
                    config=TextStepConfig(
                        prompt=(
                            "Review and polish the following analysis "
                            "for accuracy, clarity, and completeness.\n\n"
                            "Topic: France's population and its share "
                            "of the EU population.\n"
                            "Key data: France 68M (INSEE 2024), "
                            "EU 448M (Eurostat 2024), ratio ~15.18%.\n\n"
                            "Produce a concise, authoritative final "
                            "answer. Output only the polished text."
                        ),
                        system_prompt=(
                            "You are a senior editor. Review for factual "
                            "accuracy and clarity. Output only the final "
                            "polished answer."
                        ),
                        model="gpt-4o-mini",
                        temperature=0.3,
                        max_tokens=500,
                    ),
                ),
            ],
            constraints=SkillConstraints(timeout_seconds=60.0, max_retries=1),
        )

    async def interpret(self, result: Any) -> SkillResult:
        if not isinstance(result, OrchestratorResult):
            return SkillResult(success=False, error_message="Unexpected result type")

        if not result.success:
            return SkillResult(success=False, error_message="Execution failed")

        # Prefer the review step, fall back to draft
        for step_id in ("review", "draft"):
            step = result.step_results.get(step_id)
            if step and step.ok:
                return SkillResult(
                    output=step.value,
                    success=True,
                    provenance=SkillProvenance(skill_name=self.name),
                )

        return SkillResult(success=False, error_message="No answer produced")


# ╔══════════════════════════════════════════════╗
# ║  Wire Everything Together                    ║
# ╚══════════════════════════════════════════════╝


async def run_agent(question: str) -> None:
    # ── 1. Load config ────────────────────────
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"{RED}Error: OPENAI_API_KEY not found in .env{RESET}")
        sys.exit(1)

    # ── 2. EventBus (terminal progression) ────
    bus = setup_event_bus()

    # ── 3. Handlers ───────────────────────────
    text_handler = TrackedOpenAITextHandler(
        api_key=api_key, model="gpt-4o-mini", event_bus=bus,
    )

    fn_handler = ToolkitFunctionHandler(event_bus=bus)
    fn_handler.register("lookup", lookup)
    fn_handler.register("calculate", calculate)

    handler_registry = HandlerRegistry()
    handler_registry.register(EventKind.TEXT, text_handler)
    handler_registry.register(EventKind.FUNCTION, fn_handler)

    # ── 4. Orchestrator wiring ────────────────
    config = OrchestratorConfig(
        default_timeout_seconds=60.0,
        default_max_retries=1,
    )
    platform = PlatformRef(registry=handler_registry, config=config)
    adapter_registry = AdapterRegistry.with_defaults()
    state = ExecutionState()

    orchestrator = DAGOrchestrator(
        adapter_registry=adapter_registry,
        platform=platform,
        state=state,
        config=config,
    )

    # ── 5. Skill -> Plan ─────────────────────
    skill = HelpfulAssistantSkill()
    input_data = UserQuestion(text=question)
    ctx = SkillContext()

    await bus.emit(AgentStarted(question=question))

    header("Phase 1: Planning")
    plan = await skill.plan(input_data, ctx)
    print(f"  Skill: {BOLD}{skill.name}{RESET}")
    print(f"  Steps: {len(plan.steps)}")
    for i, s in enumerate(plan.steps):
        deps = f" <- [{', '.join(s.depends_on)}]" if s.depends_on else ""
        icon = ">" if s.kind == "function" else "*"
        print(f"    {DIM}{i + 1}.{RESET} [{icon}] {BOLD}{s.id}{RESET} ({s.kind}){deps}")

    # ── 6. Plan -> DAG -> Execute ─────────────
    adapter = SkillToDAGAdapter()
    dag = adapter.to_dag(plan)
    print(f"\n  PlanDAG: {dag.node_count()} nodes, "
          f"topology: {' -> '.join(dag.topological_order())}")

    header("Phase 2: Execution")
    exec_dag = await orchestrator.run(dag, {"question": question})

    # ── 7. Results -> Interpret ───────────────
    header("Phase 3: Interpretation")
    orch_result = ResultAdapter().from_result(exec_dag)
    skill_result = await skill.interpret(orch_result)

    # ── 8. Completion event ───────────────────
    await bus.emit(
        AgentCompleted(
            success=skill_result.success,
            answer=str(skill_result.output or skill_result.error_message)[:80],
        )
    )

    # ── 9. Display final answer ───────────────
    header("Final Answer")
    if skill_result.success:
        for line in _wrap(str(skill_result.output), 70):
            print(f"  {line}")
    else:
        print(f"  {RED}{skill_result.error_message}{RESET}")

    # ── 10. Execution summary ─────────────────
    print()
    entries = exec_dag.entries()
    completed = [e for e in entries if e.result is not None]

    header("Execution DAG")
    for entry in entries:
        if entry.result is None:
            continue
        color = GREEN if entry.status.value == "success" else RED
        print(
            f"  {color}*{RESET} {entry.node_id:18s} "
            f"{color}{entry.status.value:10s}{RESET}  "
            f"v{entry.state_version or '?'}"
        )

    print(f"\n{DIM}  Steps executed: {len(completed)}")
    print(f"  All succeeded:  {orch_result.success}")
    print(f"  State snapshots: {state.version_count}{RESET}\n")


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
        else "What is the population of France and what percentage "
             "of the EU population does it represent?"
    )
    asyncio.run(run_agent(question))
