# auditable-agent

An enterprise-style LLM agent that is **reliable, observable, and auditable by
construction** — built to answer the question teams actually ask before
shipping agents: *how do you know it works?*

Most agent demos show a happy path. This project instead engineers (and
proves, with an eval suite) the three things production agents need:

1. **Evaluation** — a scenario suite that checks *real side effects* (was the
   ticket actually opened? was the email actually sent?), not just the
   agent's final words. Metrics: success rate, steps, tool errors, format
   retries, latency. Safety scenarios assert the agent **refuses** unsafe
   requests and is **honest** about missing data.
2. **Observability** — every run writes a replayable JSONL trace: each
   thought, tool call, observation, error, and latency, one span per line.
3. **Recovery** — malformed model output is schema-validated and retried;
   tool failures (including injected transient outages) are fed back as
   observations so the agent re-plans instead of crashing.

## Runs on local models — $0 API cost

The client speaks the OpenAI-compatible API, so any of these work:

| Provider | `AGENT_BASE_URL` |
|---|---|
| LM Studio | `http://localhost:1234/v1` |
| Ollama | `http://localhost:11434/v1` |
| llama.cpp server | `http://localhost:8080/v1` |
| any cloud OpenAI-compatible endpoint | its `/v1` URL |

Tool calls use **prompted JSON + schema validation + bounded retries** rather
than native function-calling, so small on-device models (e.g. Qwen3 4B/8B)
work — and graceful handling of imperfect output is part of the design, not
an accident.

## Quickstart

```bash
pip install -r requirements.txt
pytest                      # deterministic tests via MockLLM — no model needed

# point at a local model (example: LM Studio serving Qwen3-4B)
export AGENT_BASE_URL=http://localhost:1234/v1
export AGENT_MODEL=qwen3-4b-instruct

python -m evals.run_evals --repeats 3   # run the eval suite, write evals/report.md
```

Traces land in `traces/<run_id>.jsonl` — one JSON span per step.

## Architecture

```
task ─► plan → act → observe loop (agent/loop.py)
              │ prompted-JSON actions, schema-validated, bounded retries
              ├─► ToolRegistry (agent/tools/) — mock email · calendar · CRM
              │     in-memory state ⇒ evals assert real side effects
              │     fault injection ⇒ recovery is tested, not assumed
              └─► Tracer (agent/tracing.py) — replayable JSONL spans
evals/ ─ scenario suite (YAML) ─► success/steps/errors/latency ─► report.md
```

## The mock workplace

Six tools over an in-memory workplace (inbox, calendar, CRM, tickets):
`search_email`, `send_email`, `list_events`, `create_event`,
`lookup_customer`, `create_ticket`. Mutations are recorded, so an eval can
assert "a high-priority ticket now exists for dev@acme.com" — the agent
can't pass by just *saying* it did the work.

## Eval scenarios (evals/scenarios.yaml)

| scenario | what it proves |
|---|---|
| triage_webhook_outage | multi-step: search → CRM lookup → ticket with right priority |
| renewal_pricing_reply | acting on the right thread, real email sent |
| schedule_around_conflict | checks calendar first, avoids a double-booking |
| survives_transient_outage | tool fails once (injected) — agent retries and succeeds |
| honest_about_missing_data | says "not found" instead of hallucinating a customer |
| refuses_unsafe_request | declines to mass-email credit-card numbers |
