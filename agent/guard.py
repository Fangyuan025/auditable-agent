"""Prompt-injection defenses — enforced, measurable, independently toggleable.

The v0.3 eval campaign showed that when the user's own task points at a
poisoned email ("find the urgent system notice and handle it"), every model
at every size followed the attacker's embedded instruction. Scale does not
fix this, so the harness must. Two layers, each measurable on its own:

1. **Provenance tags** (prompt-layer, advisory): observations from tools
   that return external content (inbox bodies) are wrapped in an EXTERNAL
   DATA marker — the model is reminded that quoted content is data, never
   instructions. Toggle: AGENT_PROVENANCE_TAGS. Applied in agent/loop.py.

2. **Egress guard** (tool-layer, enforced — this module): outbound actions
   (send_email, calendar invites) are only allowed to recipients that appear
   in the *user's task* or in the *trusted CRM directory*. An address that
   exists only inside untrusted content — like attacker@evil.com, which
   occurs nowhere but the poisoned email body — is blocked with a
   PolicyError the model observes and can recover from. A compromised model
   can still be fooled; it can no longer exfiltrate. Toggle: AGENT_EGRESS_GUARD.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


class PolicyError(Exception):
    """An action the harness refuses to execute (distinct from tool failure)."""


@dataclass
class EgressGuard:
    """Recipient allowlist derived from trusted provenance only.

    Trusted sources: the user's own words (task text) and the CRM — an
    internal store the attacker cannot write to. Untrusted content (email
    bodies) never adds trust, which is exactly what breaks the exfiltration
    step of an injection attack.
    """
    enabled: bool = True
    trusted: set[str] = field(default_factory=set)

    def trust_text(self, text: str) -> None:
        """Trust every address the user themselves wrote."""
        self.trusted |= {a.lower() for a in EMAIL_RE.findall(text)}

    def check(self, tool_name: str, recipient: str, directory: set[str]) -> None:
        """Raise PolicyError unless `recipient` has trusted provenance."""
        if not self.enabled:
            return
        r = recipient.lower().strip()
        if r in self.trusted or r in {d.lower() for d in directory}:
            return
        raise PolicyError(
            f"outbound '{tool_name}' to '{recipient}' blocked: address appears "
            f"neither in the user's request nor in the CRM directory. If the "
            f"user really wants this, they must name the recipient explicitly. "
            f"Do not retry with the same recipient.")
