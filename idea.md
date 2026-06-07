# Probe-Routed Multi-Agent Orchestration — Decision Log

**Event:** WeaveHacks 4 — Multi-Agent Orchestration (June 6–7, 2026, W&B SF)
**One-liner:** A learned **router** that uses linear probes on a small model's internal states to dispatch queries to the right agent and tools — cheaper and more accurate than stuffing the whole catalog into context.

---

## 1. The thesis

The standard agent loads all N tool schemas into context and reasons over them. That breaks on **token cost** and **accuracy** as N grows. We replace the in-context search with linear probes trained on the model's hidden states, so the agent only ever sees a small, pre-screened shortlist.

The probe is a **filter in front of the context window**, not a replacement for the agent's judgment. The agent still reasons and picks — it just picks from 5 candidates instead of 200.

---

## 2. Core mechanism (settled)

- Use the LLM as a **frozen feature extractor**. We never decode its output.
- Flow: `query → forward pass → hidden state at layer L (last token) → LogReg → label`.
- The hidden state is just a vector of numbers; the probe is a linear map from that vector to a tool/agent.
- **Hallucination is not a risk here** — there's no generation. The probe can only output a label it was trained on. "Does it work" = held-out accuracy, a number we measure, not a hope.

### Locked details
- **Train/inference must match exactly:** same layer, same token position, same *query-only* forward pass (no tools in context). Mismatch here is the #1 silent failure.
- **Labels come from ground truth, never from the model grading itself.** Self-labeling caps us at the baseline we're trying to beat. Generate labels synthetically from templated per-tool scenarios (intended routing known by construction) or a one-time oracle pass.
- **Don't hardcode layer 27** — sweep layers and pick by validation accuracy (often upper-middle of the stack). The accuracy-by-layer plot is a good slide.

---

## 3. Architecture — two-probe hierarchical routing (current design)

One ~1.5B model, two linear probes on its hidden state.

1. **Agent probe** → shortlist **top 2 agents**. Model picks 1.
2. **Tool probe** (masked to the chosen agent) → shortlist **top 5 tools**. Agent reasons over the 5 and executes.

### Key principles
- **Shortlist, don't decide — at both levels.** The probe only narrows; the model/agent makes the final pick. A probe near-miss is recoverable as long as the right option is in the shortlist.
- **Masking:** the tool probe scores all ~200 tools, zeros out everything not owned by the chosen agent, then takes the argmax over what's left. A flat 200-way problem becomes an easy ~10-way decision at inference.
- **Tool tags do double duty:** each tool is tagged with its owning agent → the tag *is* the mask, and (read in reverse) the winning tool's tag tells you the agent.
- **External probe:** the probe is a module invoked *between* steps, not baked into the model. This is what enables multi-step.
- **Multi-step loop:** after an agent executes, the result updates the context, and we **re-probe on the new context** for the next hop.
- **One forward pass per hop feeds both probes.** The agent choice only changes the *mask* the tool probe uses — it does not require a new forward pass. We re-run the forward pass only when the context genuinely changes (between hops).

### `k` values are dials, not constants
- Agent shortlist = 2, tool shortlist = 5 — both are recall-vs-cost knobs.
- Set each from the eval set: plot recall@k vs token cost and pick the knee. Optionally make k adaptive (more candidates when probe confidence is spread, fewer when peaked).

---

## 4. Why this beats RAG (have the answer ready)

- RAG retrieves by **similarity**; the probe selects by **supervision** (a learned decision boundary). The correct tool often isn't the most textually similar one.
- The probe reads the **executing model's own representation** of the query (post coreference/negation/intent), not a separate embedding model's surface view.
- It doesn't depend on how well tool descriptions are written.
- **Honest tradeoff:** the probe is closed-set — new tools need a (cheap, offline) relabel + retrain; RAG is zero-shot/open-set and scales to thousands trivially.
- **Hybrid (roadmap):** RAG shortlists the open world → probe picks precisely within the known set. Strongest version of the "agent bazaar."

---

## 5. Mechanistic interpretability framing (the depth story)

Three rungs from correlational to mechanistic:

1. **Linear probe** — observational / correlational. *(what we have)* Justified by the linear representation hypothesis.
2. **Activation patching / steering** — perturb the probe direction and watch the route flip → proves it's **causal**. *(makes it "mechanistic")*
3. **Sparse autoencoder features** (e.g., **Gemma Scope**, pretrained — don't train your own) — name the features behind each route → **explainable, auditable routing**.

Honesty guardrail: SAE features are only *seemingly* interpretable; treat feature names as hypotheses, not ground truth.

---

## 6. Scale & tooling

- ~10 subagents, ~200 tools total. **3–6 real integrations** (e.g., reputation lookup, log/SIEM search, Slack alert, IP/geo); the rest realistic **mock stubs** (fine — routing accuracy depends on labels, not execution).
- Base model: **Qwen 1.5B** currently (open weights required for activations). Switch to **Gemma 2 + Gemma Scope** only if we pursue the SAE rung.
- Compute: **Modal** (hosted GPU).
- Single-probe alternative (faster to ship): one global probe over all 200 tools, masked to the chosen agent, agent read off the winning tool's tag. Two-probe is cleaner to *explain*; one-probe is faster to *build*.

---

## 7. Use case — SOC (security operations) triage desk

- Domain skin: a fleet of security specialists (alert triage, log search, threat intel, malware/file analysis, IP/domain reputation, incident response, access review...). "Simulating attacks" = a live stream of varied events that must be routed correctly.
- Good fit because the fleet is **large, known, and stable** — the probe's sweet spot.
- **Honesty boundary (critical):** the system **routes / triages**; it does **not** detect or block attacks. Claim: *"dispatch the correct security agent + tools faster and cheaper than a frontier orchestrator, at equal-or-better routing accuracy."* Do not let the UI imply we stop breaches.

---

## 8. Demo plan

- **Side-by-side:** frontier model doing in-context orchestration over all 200 tools vs. our small-model + probe router.
- Comparison axis: **accuracy / recall, token cost, latency, $** — not raw capability.
- Attack-simulation feed on top; **benchmarks below**.
- Make the infrastructure benefit *visible*: live token/latency counters, the moment the baseline routes wrong and we route right, the leaderboard.

---

## 9. Weave usage (required to win; this is the centerpiece)

- **Trace tree** of each run: `orchestrator → agent probe → tool probe → agent execution`, every node logging probe confidence (and SAE features once added).
- **Evaluations (priority):** a dataset of `(event → correct agent, correct tools)`; scorers for agent recall@2, tool recall@5, plus tokens/latency/cost. Run across **frontier-in-context vs RAG vs probe router** → one leaderboard. Log the frontier baseline's choices + cost too (apples-to-apples).
- Sponsor fits: **Redis** = agent/tool registry + vector store for the RAG baseline; **CopilotKit** = the live routing cockpit.

---

## 10. Metrics funnel (log each stage separately)

`agent recall@2  →  model picks agent  →  tool recall@5  →  agent picks tool  →  execute`

End-to-end accuracy is a product of these stages. A miss can come from any one — log them separately so we know which to fix.

---

## 11. Build order (NOT one-shot)

Thin vertical slice first, then deepen. Each stage is independently demoable.

1. **Agent probe + held-out accuracy number.** The core risk and the core claim, in isolation. Build this first; if accuracy is bad, find out in hour 3, not hour 20.
2. **Minimal end-to-end:** query → agent probe → route → one agent runs, with Weave tracing on.
3. **Tool probe + masking + top-5.** Full architecture.
4. **Frontier baseline + Weave Evaluation leaderboard.** The benchmark that actually wins — prioritize over polish.
5. **Attack-sim UI.** Last. It's the make-up, not the skeleton.

---

## 12. Risks & caveats to own

- **Label quality / dataset is the long pole**, not the probes. Generate synthetically per agent/tool.
- **Data sizing:** size the training set by *examples per tool* (dozens–100+ each), not by tool count.
- **Closed-set:** new tools/agents need a relabel + retrain (cheap, offline).
- **Latency is roughly a wash** vs a single big in-context call (more round-trips). The win is **cost + accuracy** — lead with those.
- **Error compounding** in multi-step → the shortlist-not-decide pattern and confidence thresholds are load-bearing, not optional.
- A router is infrastructure → it loses to flashy apps *unless the benefit is made visible* in the demo.

---

## 13. Still open / to decide

- One global probe vs. two probes (ship speed vs. explanation clarity).
- Whether to add the SAE rung (→ implies Gemma 2 + Gemma Scope).
- Final `k` values for agent and tool shortlists (set from recall-vs-cost curves).
- Which 3–6 integrations are real.