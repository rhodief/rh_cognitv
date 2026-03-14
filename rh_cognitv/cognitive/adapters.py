"""
L1→L2 Adapter — translate SkillPlan ↔ PlanDAG / ExecutionDAG ↔ OrchestratorResult.

The adapter is the boundary between the Cognitive Layer (L1) and the
Orchestrator Layer (L2). It knows both type systems; no other cognitive
module imports from L2.

Phase 3.6 — L1→L2 Adapter.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rh_cognitv.cognitive.models import (
    DataStepConfig,
    FunctionStepConfig,
    SkillPlan,
    SkillStep,
    TextStepConfig,
    ToolStepConfig,
)
from rh_cognitv.orchestrator.execution_dag import ExecutionDAG
from rh_cognitv.orchestrator.models import NodeResult
from rh_cognitv.orchestrator.nodes import (
    DataNode,
    ExecutionNode,
    FunctionNode,
    TextNode,
    ToolNode,
)
from rh_cognitv.orchestrator.plan_dag import DAGBuilder, PlanDAG


# ──────────────────────────────────────────────
# OrchestratorResult
# ──────────────────────────────────────────────


class OrchestratorResult(BaseModel):
    """Adapter-normalized result for Skill.interpret().

    Maps step IDs from the original SkillPlan to their NodeResult
    values from the ExecutionDAG.
    """

    step_results: dict[str, NodeResult] = Field(default_factory=dict)
    success: bool = True


# ──────────────────────────────────────────────
# SkillToDAGAdapter
# ──────────────────────────────────────────────


class SkillToDAGAdapter:
    """Translate SkillPlan → PlanDAG.

    Maps each SkillStep to the corresponding L2 node type,
    wires edges, serializes context_refs into ext, and applies
    plan-level constraints to each node.
    """

    def to_dag(self, plan: SkillPlan) -> PlanDAG:
        """Convert a SkillPlan into a PlanDAG via DAGBuilder.

        Steps are mapped to L2 nodes by kind. Edges come from
        each step's depends_on list; steps with no explicit
        dependencies are wired sequentially after the previous step.
        """
        builder = DAGBuilder(plan.name)

        for step in plan.steps:
            node = self._make_node(step, plan)
            builder.add_node(step.id, node)

        for i, step in enumerate(plan.steps):
            if step.depends_on:
                for dep_id in step.depends_on:
                    builder.edge(dep_id, step.id)
            elif i > 0:
                builder.edge(plan.steps[i - 1].id, step.id)

        return builder.build()

    def _make_node(self, step: SkillStep, plan: SkillPlan) -> ExecutionNode:
        """Map a SkillStep to the matching L2 ExecutionNode."""
        ext: dict = {}
        if step.context_refs:
            ext["context_refs"] = [ref.model_dump() for ref in step.context_refs]

        timeout = plan.constraints.timeout_seconds if plan.constraints else None
        retries = plan.constraints.max_retries if plan.constraints else None

        common = dict(
            id=step.id,
            timeout_seconds=timeout,
            max_retries=retries,
            ext=ext,
        )

        cfg = step.config
        if step.kind == "text" and isinstance(cfg, TextStepConfig):
            return TextNode(
                prompt=cfg.prompt,
                system_prompt=cfg.system_prompt,
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                **common,
            )
        if step.kind == "data" and isinstance(cfg, DataStepConfig):
            return DataNode(
                prompt=cfg.prompt,
                output_schema=cfg.output_schema,
                model=cfg.model,
                **common,
            )
        if step.kind == "function" and isinstance(cfg, FunctionStepConfig):
            return FunctionNode(
                function_name=cfg.function_name,
                args=cfg.args,
                kwargs=cfg.kwargs,
                **common,
            )
        if step.kind == "tool" and isinstance(cfg, ToolStepConfig):
            return ToolNode(
                prompt=cfg.prompt,
                tools=cfg.tools,
                model=cfg.model,
                **common,
            )
        raise ValueError(
            f"Cannot map step kind={step.kind!r} with config type={type(cfg).__name__}"
        )


# ──────────────────────────────────────────────
# ResultAdapter
# ──────────────────────────────────────────────


class ResultAdapter:
    """Translate ExecutionDAG → OrchestratorResult.

    Extracts NodeResult values by node ID from the ExecutionDAG
    for consumption by Skill.interpret().
    """

    def from_result(self, execution_dag: ExecutionDAG) -> OrchestratorResult:
        """Build an OrchestratorResult from a completed ExecutionDAG."""
        step_results: dict[str, NodeResult] = {}

        for entry in execution_dag.entries():
            if entry.result is not None:
                step_results[entry.node_id] = entry.result

        success = all(r.ok for r in step_results.values()) if step_results else True
        return OrchestratorResult(step_results=step_results, success=success)
