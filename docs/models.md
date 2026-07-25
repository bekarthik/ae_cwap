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

## Capabilities, and why the canvas changes shape

Backends do not accept the same knobs, and the differences are not cosmetic:

| Knob | Claude | Open models via OpenAI-compatible |
| --- | --- | --- |
| `effort` | supported | not a concept |
| `temperature` | **rejected with a 400** | supported |
| `max_tokens` | supported | supported |

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
- **Keep prompts explicit.** The scaffolder generates prompts that name each
  input plainly, which smaller models follow more reliably than terse ones.
- **The stub is still there.** `CWAP_LLM_PROVIDER=stub` runs the whole platform
  with no model at all, which is what the test suite uses and what makes the
  execution tests assert on exact outputs.

---

## Verifying a backend

```bash
curl -s localhost:8000/health                       # provider and model in use
curl -s localhost:8000/api/runtime -H "Authorization: Bearer $TOKEN" | jq
```

`/api/runtime` reports the resolved provider, model, endpoint, capability flags,
the embedding configuration, and — if the deployment is misconfigured — the exact
setting to fix. It reports that as data rather than raising, so a bad setting
shows an actionable message in the UI instead of a 500 on an unrelated page.
