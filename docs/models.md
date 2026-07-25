# Model backends

The platform is model-agnostic by construction. `services/llm_proxy` is the only
module that talks to a model, and it exposes one narrow interface, so a workflow
authored against a local Llama runs unchanged against a hosted Claude and back
again. Nothing in a saved graph names a vendor.

---

## Choosing one

Set `CWAP_LLM_PROVIDER`. Everything else has a sensible default.

### Open-weight models on your own machine

No credentials, no account, no network egress.

```bash
# Ollama — https://ollama.com
ollama pull llama3.1
export CWAP_LLM_PROVIDER=ollama
export CWAP_LLM_MODEL=llama3.1
```

```bash
# vLLM, LM Studio, llama.cpp, TGI — same idea, different default port
export CWAP_LLM_PROVIDER=vllm        # or lmstudio / llamacpp / tgi
export CWAP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

| Preset | Default endpoint | Needs a key |
| --- | --- | --- |
| `ollama` | `http://localhost:11434/v1` | no |
| `vllm` | `http://localhost:8000/v1` | no |
| `lmstudio` | `http://localhost:1234/v1` | no |
| `llamacpp` | `http://localhost:8080/v1` | no |
| `tgi` | `http://localhost:8080/v1` | no |
| `litellm` | `http://localhost:4000/v1` | no |

### Hosted APIs

```bash
export CWAP_LLM_PROVIDER=together        # open-weight models, hosted
export CWAP_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
export CWAP_LLM_API_KEY=...
```

| Preset | Default endpoint |
| --- | --- |
| `openai` | `https://api.openai.com/v1` |
| `together` | `https://api.together.xyz/v1` |
| `groq` | `https://api.groq.com/openai/v1` |
| `openrouter` | `https://openrouter.ai/api/v1` |
| `fireworks` | `https://api.fireworks.ai/inference/v1` |
| `deepseek` | `https://api.deepseek.com/v1` |
| `mistral` | `https://api.mistral.ai/v1` |

### Anthropic

```bash
export CWAP_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...       # or run `ant auth login`
export CWAP_LLM_MODEL=claude-opus-5
```

### Anything else

```bash
export CWAP_LLM_PROVIDER=openai_compatible
export CWAP_LLM_BASE_URL=https://my-gateway.internal/v1
export CWAP_LLM_MODEL=whatever-it-serves
```

`CWAP_LLM_BASE_URL` overrides any preset, so `ollama` pointed at
`http://gpu-box.internal:11434/v1` works exactly as you would expect.

---

## Why only two client implementations

Almost every model server in use today speaks the OpenAI chat-completions wire
format, so `OpenAICompatibleProvider` covers open-weight models on a laptop,
self-hosted clusters, and a dozen hosted APIs with one client. Anthropic gets its
own implementation because its API genuinely differs.

Adding a backend is an entry in `services/llm_proxy/presets.py`, not a new class.

The OpenAI-compatible client is written directly on httpx rather than a vendor
SDK. That is deliberate: the servers it must satisfy agree on the wire format but
differ at the edges — some omit `usage`, some return a bare error string, some
put the answer in `reasoning_content`. Owning the request and the error mapping
is what lets a local model produce the same clear, actionable failures as a
hosted one, and it means someone running entirely offline installs nothing extra.

---

## Picking one from the product

Click the model chip in the canvas header. It lists every backend the platform
knows — grouped into "on your own hardware" and "hosted" — the models each
commonly serves, and what each of those can do: tool calling, vision, a reasoning
mode. Selecting one shows the exact settings to apply.

It shows those settings rather than applying them. The provider is process-wide
server configuration, and a browser session switching model underneath other
people's in-flight runs is not something a page should be able to do.

The catalogue in `services/llm_proxy/catalogue.py` is a hint, never a gate. An
unlisted model still runs — so a private fine-tune is one keystroke away in the
model field.

**Tool support is never inferred from a model's name.** That rule exists because
the two possible mistakes are not symmetric:

| Guess | If wrong | Cost |
| --- | --- | --- |
| "this model has tools" | The server rejects the request | One retried call, then a permanent downgrade |
| "this model has no tools" | It did have them | The slower prompted protocol **forever**, with no signal that would ever correct it |

So a catalogued model reports what its entry says, anything else is assumed
capable, and the first real request settles it. An entry may only assert the
negative when the *provider documents it* — currently just `deepseek-reasoner`.
Vision and thinking are still guessed from a name, for the same reason tools are
assumed rather than denied: the guess is recoverable. A wrong thinking guess
sends one parameter the server does not take and is corrected by the retry
described below; a wrong vision guess only changes which controls are offered.

The canvas reports where its answer came from, and only states a limitation it
did not guess:

| `tool_support` | Meaning | Shown to the user? |
| --- | --- | --- |
| `observed` | This server rejected a tools request | Yes — "the model rejected a tool-calling request" |
| `catalogue` | The model is listed and its entry says so | Yes — "does not support native tool calling" |
| `configured` | An operator set `CWAP_LLM_TOOL_MODE=prompted` | Yes — as a deployment setting, not a model limitation |
| `assumed` | Nobody has checked yet | **No** |

---

## Tool calling, and agents on models that lack it

An agent loop needs exactly one capability the rest of the platform does not:
tool calling. Plenty of capable open-weight models never learned it, and whether
a given server supports it is **not discoverable up front** — a hosted API
advertises it, a local llama.cpp build may or may not have it, and there is no
endpoint to ask.

So the first agent turn simply tries:

```
converse(tools=[…])
   │
   ├── 200 → native tool calling. Done.
   │
   └── 4xx that is *about* tools and says "unsupported"
          │
          ├── downgrade this provider permanently
          └── retry immediately on the prompted protocol   ← the turn still succeeds
```

The prompted protocol describes the tools in the system prompt and asks for a
single JSON object. Anything that does not parse as a call is treated as a final
answer, so a model that ignores the protocol degrades to a plain response rather
than erroring, and a hallucinated tool name is never dispatched.

The rejection match is subject-plus-negation ("mentions tools" **and** "says
unsupported") rather than a list of exact sentences, because every backend
phrases it differently: *does not support tools*, *tools are unsupported*,
*unknown field: tools*, *extra inputs are not permitted*. It is deliberately
strict on the subject: mistaking a rejected API key for a missing capability
would silently degrade every agent in the deployment and hide the real problem.

`CWAP_LLM_TOOL_MODE` forces the decision when you already know the answer:

| Value | Behaviour |
| --- | --- |
| `auto` (default) | try native, downgrade permanently on a tool rejection |
| `native` | always send `tools`; a rejection surfaces as an error |
| `prompted` | never send `tools`; use the JSON protocol from the start |

---

## Reasoning models

A model that can think and is never asked to is an expensive model used badly.
Detecting the capability and then sending a plain request is worse than not
detecting it: the canvas says *thinking* while nothing about the call differs.

The obstacle is that there is no agreed way to ask. The OpenAI wire format is
near-universal for *messages* and completely unstandardised for *reasoning* —
every server invented its own parameter, and sending the wrong one is a 400:

| Backend | Parameter |
| --- | --- |
| OpenAI, LM Studio, Groq, Fireworks, LiteLLM, Together, Mistral | `reasoning_effort` |
| OpenRouter | `reasoning: {effort}` |
| Ollama | `think: true` |
| vLLM, llama.cpp, TGI | `chat_template_kwargs: {enable_thinking: true}` |
| DeepSeek reasoner | *nothing* — it reasons unconditionally, and asking is a 400 |

`services/llm_proxy/reasoning.py` holds that table, and it is the whole extension
point: a new backend is a row, not a provider class. A rejection is handled the
same way a rejected tools request is — drop the parameter, retry immediately,
stop asking:

```
complete/converse with the reasoning parameter
   │
   ├── 200 → the model thought. Keep what it thought.
   │
   └── 4xx that is *about* reasoning and says "unsupported"
          │
          ├── stop sending it on this provider (effort stops being offered)
          └── retry immediately without it   ← the turn still succeeds
```

Two things follow from thinking being real rather than decorative:

* **`effort` reaches open models now.** A thinking-capable model on a backend
  with a depth parameter reports `supports_effort`, so the depth control on the
  canvas does something instead of being reported as ignored. When the server
  rejects the parameter, `supports_effort` goes false — but `supports_thinking`
  stays true, because what the server refused was the *knob*, not the capability.
* **The output budget grows.** Reasoning is spent from the same budget as the
  answer, so a thinking model against a 4k ceiling can spend the whole allowance
  thinking and return a truncated sentence. Requests that engage thinking get at
  least 8192 tokens.

What the model thought is **kept, not discarded**. It arrives in
`reasoning_content`, `reasoning` or `thinking` depending on the server (and as
`thinking` content blocks on Claude), and lands in three places: an
`agent.reasoned` line in the live log, the full text on the step's context, and
— when a model spends its whole budget thinking and returns no answer — as the
step's output, which beats reporting silence.

One deliberate exclusion: on the prompted tool protocol, a tool call is only
read out of the *answer*, never out of the reasoning. A thinking model drafts and
discards candidate calls while working, and invoking one of those would run a
skill the model had already decided against.

`CWAP_LLM_THINKING_MODE=off` turns the whole thing off deployment-wide.

---

## Capabilities, and why the canvas changes shape

Backends do not accept the same knobs, and the differences are not cosmetic:

| Knob | Claude | Open models via OpenAI-compatible |
| --- | --- | --- |
| `effort` | supported | on thinking models whose server takes a depth parameter |
| `temperature` | **rejected with a 400** | supported |
| `max_tokens` | supported | supported |
| tool calling | supported | **varies by model** — see above |
| vision | model-dependent | model-dependent |

Pretending otherwise would mean either sending a parameter that errors, or
silently dropping one the user set. The platform does neither:

1. A provider declares `ProviderCapabilities`.
2. `GET /api/runtime` reports them, so the Inspector renders an effort selector
   against Claude and a temperature slider against Llama.
3. A node keeps knobs meant for *other* backends — that is what makes a workflow
   portable — and the Inspector labels them inactive rather than hiding them.
4. At run time, anything the backend cannot honour is returned in
   `ignored_options`, logged as a `WARN`, and recorded on the step. A run report
   never implies a setting took effect when it did not.

---

## Embeddings

RAG needs a model too, and the same reasoning applies.

```bash
# Offline default — lexical, deterministic, no credentials
CWAP_EMBEDDING_PROVIDER=hashing

# A real embedding model, local
CWAP_EMBEDDING_PROVIDER=ollama
CWAP_EMBEDDING_MODEL=nomic-embed-text

# Or hosted
CWAP_EMBEDDING_PROVIDER=openai
CWAP_EMBEDDING_MODEL=text-embedding-3-small
CWAP_EMBEDDING_API_KEY=...
```

The embedding endpoint is configured independently of the chat model, because
they are frequently different services — a local `nomic-embed-text` alongside a
hosted chat model is a normal setup.

**Changing the embedding model invalidates existing corpora.** Cosine distance
between vectors from two different embedding spaces produces scores that look
plausible and mean nothing. Every corpus records which embedder indexed it, and
retrieval refuses to compare across models with a message telling you to
re-upload. Silent nonsense would be far worse than a clear error.

Batching: `RemoteEmbedder.embed_batch` sends one request per document rather than
one per chunk. Against a local embedding server the difference is minutes.

---

## Practical notes for local models

- **Raise the timeout.** `CWAP_LLM_TIMEOUT` defaults to 120s. A 70B model on
  modest hardware will exceed that; the error message says so when it happens.
- **Lower `max_tokens` on nodes.** Small models are slower per token, and a
  truncated answer is flagged in the run report (`finish_reason: length`) rather
  than passed downstream as if complete.
- **Keep prompts explicit.** The designer generates objectives that name each
  input plainly, which smaller models follow more reliably than terse ones.
- **Prefer a model with native tool calling for agent steps.** The prompted
  fallback works everywhere, but a model trained on tool use follows a multi-step
  plan more reliably. `qwen2.5` and `llama3.1` both have it and both run on a
  laptop.
- **Lower the iteration budget on small models.** An agent's budget is its
  ceiling on model calls; six iterations of a 70B model on modest hardware is a
  long wait. A budget-exhausted agent still returns its best answer, and records
  that the budget was too small.
- **The stub is still there.** `CWAP_LLM_PROVIDER=stub` runs the whole platform
  with no model at all, which is what the test suite uses and what makes the
  execution tests assert on exact outputs.

---

## Verifying a backend

```bash
curl -s localhost:8000/health                       # provider and model in use
curl -s localhost:8000/api/runtime -H "Authorization: Bearer $TOKEN" | jq
```

`/api/runtime` reports the resolved provider, model, endpoint, capability flags
(including whether tool calling is native or prompted), the full catalogue of
providers and models the picker offers, the embedding configuration, and — if the
deployment is misconfigured — the exact setting to fix. It reports that as data rather than raising, so a bad setting
shows an actionable message in the UI instead of a 500 on an unrelated page.
