# Lokalni setup za intervju

Python okruženje je u `.venv`. Nema starter koda. Na intervjuu kucaš sam.

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
