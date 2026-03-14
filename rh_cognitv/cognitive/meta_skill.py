"""
MetaSkill — V2 stub.

Extends Skill with ``generate_skill`` and ``generate_dag`` methods
for declarative skill/DAG generation from natural-language descriptions.

V1: both methods raise ``NotImplementedError``.
V2: concrete implementations will use LLMs to generate skills and DAGs.

Phase 3.9.1 — MetaSkill Stub (DI-L1-03).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .models import SkillConfig, SkillContext, SkillPlan, SkillResult
from .protocols import MetaSkillProtocol
from .skill import Skill


class MetaSkill(Skill, MetaSkillProtocol):
    """Base class for meta-skills that generate other skills or DAGs.

    Satisfies ``MetaSkillProtocol``. Subclass and override
    ``generate_skill`` / ``generate_dag`` in V2.

    V1: both generation methods raise ``NotImplementedError``.
    The standard ``plan`` / ``interpret`` / ``name`` / ``description``
    must still be implemented by concrete subclasses.
    """

    async def generate_skill(
        self, description: str, context: SkillContext
    ) -> SkillConfig:
        """Generate a SkillConfig from a natural-language description.

        V1: raises NotImplementedError.
        """
        raise NotImplementedError(
            "MetaSkill.generate_skill is a V2 feature"
        )

    async def generate_dag(
        self, description: str, context: SkillContext
    ) -> Any:
        """Generate a PlanDAG from a natural-language description.

        Returns ``Any`` to avoid hard L2 import; the actual return
        type is ``PlanDAG`` (L2).

        V1: raises NotImplementedError.
        """
        raise NotImplementedError(
            "MetaSkill.generate_dag is a V2 feature"
        )
