"""Mock workplace tools.

Each tool = name + description + params (shown to the model) + a callable.
Backends are tiny in-memory stores so evals can assert on side effects
(e.g. "an email to X was actually sent"). The registry supports failure
injection to test the agent's error recovery.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.config import settings
from agent.guard import EgressGuard
from agent.tools.workplace import Workplace


class ToolError(Exception):
    """Raised by tools on invalid input or simulated outages."""


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # param name -> human description (shown to model)
    fn: Callable[..., Any]
    # arg that names an outbound recipient — gated by the egress guard
    egress_param: str | None = None
    # output contains untrusted external content — provenance-tagged in the loop
    external: bool = False


@dataclass
class ToolRegistry:
    workplace: Workplace = field(default_factory=Workplace)
    # tool name -> number of times it should fail before succeeding
    inject_failures: dict[str, int] = field(default_factory=dict)
    guard: EgressGuard = field(
        default_factory=lambda: EgressGuard(enabled=settings.egress_guard))

    def __post_init__(self) -> None:
        w = self.workplace
        self._tools: dict[str, Tool] = {t.name: t for t in [
            Tool("search_email", "Search the inbox by keyword.",
                 {"query": "keyword to search for"}, w.search_email,
                 external=True),
            Tool("send_email", "Send an email.",
                 {"to": "recipient address", "subject": "subject line",
                  "body": "message body"}, w.send_email,
                 egress_param="to"),
            Tool("list_events", "List calendar events on a date.",
                 {"date": "YYYY-MM-DD"}, w.list_events),
            Tool("create_event", "Create a calendar event (invites the attendee).",
                 {"date": "YYYY-MM-DD", "time": "HH:MM 24h", "title": "event title",
                  "attendee": "attendee email"}, w.create_event,
                 egress_param="attendee"),
            Tool("lookup_customer", "Look up a customer record in the CRM.",
                 {"email": "customer email address"}, w.lookup_customer),
            Tool("create_ticket", "Open a support ticket for a customer.",
                 {"customer_email": "customer email", "summary": "issue summary",
                  "priority": "low | medium | high"}, w.create_ticket),
        ]}

    def spec_for_prompt(self) -> str:
        lines = []
        for t in self._tools.values():
            args = ", ".join(f'"{k}": <{v}>' for k, v in t.params.items())
            lines.append(f"- {t.name}: {t.description} args: {{{args}}}")
        return "\n".join(lines)

    def names(self) -> list[str]:
        return list(self._tools)

    def is_external(self, name: str) -> bool:
        return name in self._tools and self._tools[name].external

    def execute(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            raise ToolError(f"unknown tool '{name}'; available: {', '.join(self._tools)}")
        if self.inject_failures.get(name, 0) > 0:
            self.inject_failures[name] -= 1
            raise ToolError(f"transient backend error in '{name}' (injected)")
        tool = self._tools[name]
        unknown = set(args) - set(tool.params)
        missing = set(tool.params) - set(args)
        if unknown or missing:
            raise ToolError(
                f"bad args for '{name}': missing={sorted(missing)} unknown={sorted(unknown)}")
        if tool.egress_param:
            # the CRM is the trusted directory: internal, attacker-unwritable
            self.guard.check(name, str(args[tool.egress_param]), set(self.workplace.crm))
        return tool.fn(**args)
