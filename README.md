# Lokalni setup za intervju

Python okruženje je u `.venv`.

## Agentic RAG skeleton

```bash
source .venv/bin/activate
python -m agent.run                                  # multi-turn REPL
python -m agent.run --trace "how do we chunk docs?"  # one turn + trace
python -m agent.run --failure-rate 1.0 "..."         # exercise the degraded path
pytest
```

Layout:

| Path | Role |
| --- | --- |
| `agent/graph.py`, `agent/routing.py` | LangGraph wiring, conditional edges, `ConversationAgent` |
| `agent/state.py` | `AgentState` channels and reducers |
| `agent/nodes/` | one file per node: prepare, analyze, tools, gather, synthesize, recovery |
| `agent/tools/` | tool contract, resilience (timeout/retry/breaker), registry, two tools |
| `agent/rag/` | simulated document store: 20-doc corpus plus a `Retriever` |
| `agent/errors.py`, `agent/observability.py` | error taxonomy, structured logs, spans, metrics |

Everything AWS-facing is simulated and marked with a `TODO` at the swap point.

Vodiči, na engleskom, da možeš da vežbaš odgovor naglas:

- `docs/01_qa_talking_points.md` — klasičan ML, transformeri, RAG, vektori, agenti, AWS
- `docs/02_system_design.md` — produkcioni RAG na AWS, uključujući uvod od oko 60 sekundi
- `docs/03_live_coding_checklist.md` — redosled za 30 minuta live codinga

## Okruženje

```bash
source .venv/bin/activate
```

Ako praviš okruženje iznova:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Instalirano je ono što intervju pominje: boto3 (Bedrock), OpenSearch klijent, LangChain, LangGraph, LangChain-AWS, Pydantic, NumPy, pytest.

Kredencijale ne commituj. Kopiraj `.env.example` u `.env` kad budeš imao ključ. Model id proveri u Bedrock konzoli pre sesije.

Interpreter u Cursoru: `.venv/bin/python`.
