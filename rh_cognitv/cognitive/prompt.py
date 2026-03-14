"""
Prompt template engine + context injection.

Provides two ways to build prompts:
- PromptBuilder: chainable programmatic API for code-based Skills (DD-L1-03 Option D)
- TemplateRenderer: str.format-based renderer for ConfigSkill templates (DD-L1-03 Option A)

Both produce BuiltPrompt(system_prompt, prompt) matching L3's TextPayload fields.

Phase 3.4.1 — Prompt Engine.
"""

from __future__ import annotations

from .models import BuiltPrompt
from .protocols import PromptProtocol


class PromptBuilder(PromptProtocol):
    """Chainable programmatic prompt builder.

    Used by code-based Skills to compose prompts from parts:
      - .system() sets the system prompt
      - .context() injects serialized context
      - .user() appends user message segments

    Builder produces BuiltPrompt(system_prompt, prompt) where prompt
    is the concatenation of context + user segments, mapping 1:1 to
    L3's TextPayload(prompt, system_prompt).

    Usage::

        prompt = (
            PromptBuilder()
            .system("You are a summarization assistant.")
            .context(serialized_memories)
            .user(f"Summarize: {input.text}")
            .build()
        )
    """

    def __init__(self) -> None:
        self._system: str | None = None
        self._context: str | None = None
        self._user_parts: list[str] = []

    def system(self, text: str) -> PromptBuilder:
        """Set the system prompt.

        Multiple calls overwrite — only the last system() value is used.
        """
        self._system = text
        return self

    def context(self, text: str) -> PromptBuilder:
        """Inject serialized context (memories/artifacts).

        Multiple calls overwrite — only the last context() value is used.
        Context is placed before user segments in the final prompt.
        """
        self._context = text
        return self

    def user(self, text: str) -> PromptBuilder:
        """Append a user message segment.

        Multiple calls accumulate — segments are joined with newlines.
        """
        self._user_parts.append(text)
        return self

    def build(self) -> BuiltPrompt:
        """Build the final prompt.

        Assembles: [context]\\n\\n[user segments joined by \\n]
        into the prompt field. Empty parts are omitted.
        """
        parts: list[str] = []
        if self._context:
            parts.append(self._context)
        if self._user_parts:
            parts.append("\n".join(self._user_parts))

        return BuiltPrompt(
            system_prompt=self._system,
            prompt="\n\n".join(parts),
        )


class TemplateRenderer:
    """str.format-based template renderer for ConfigSkill.

    Fills {input.field} and {context} placeholders using Python's
    str.format_map(). Used by ConfigSkill to auto-generate prompts
    from SkillConfig.prompt_template.

    Usage::

        renderer = TemplateRenderer()
        result = renderer.render(
            template="Summarize: {text}",
            variables={"text": "Hello world", "context": "background info"},
        )
        # result.prompt == "Summarize: Hello world"

    With system_prompt::

        result = renderer.render(
            template="Summarize: {text}",
            variables={"text": "Hello world"},
            system_prompt="You are a summarizer.",
        )
    """

    def render(
        self,
        template: str,
        variables: dict[str, str] | None = None,
        system_prompt: str | None = None,
        system_template: str | None = None,
    ) -> BuiltPrompt:
        """Render a template with variable substitution.

        Args:
            template: The prompt template with {field} placeholders.
            variables: Dict of variable names to values for substitution.
            system_prompt: Static system prompt (used as-is).
            system_template: System prompt template with {field} placeholders.
                If both system_prompt and system_template are given,
                system_template takes precedence.

        Returns:
            BuiltPrompt with rendered prompt and optional system_prompt.
        """
        safe_vars = variables or {}
        prompt = template.format_map(_SafeFormatDict(safe_vars))

        final_system: str | None = system_prompt
        if system_template is not None:
            final_system = system_template.format_map(_SafeFormatDict(safe_vars))

        return BuiltPrompt(
            system_prompt=final_system,
            prompt=prompt,
        )


class _SafeFormatDict(dict):
    """Dict subclass that returns '{key}' for missing keys.

    Prevents KeyError on unused placeholders — missing variables
    are left as-is in the template rather than raising.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
