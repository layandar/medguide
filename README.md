# MedGuide

Photo of a medicine box -> confirm the name -> grounded instructions -> spoken aloud.

```
photo ──> /api/identify (vision model via OpenRouter)
              │  user confirms / edits the name
              ▼
         /api/explain
              ├─ retrieval.py   openFDA label lookup + chunk ranking   (the RAG step)
              ├─ safety.py      allergy / interaction / pregnancy flags (plain code, not the LLM)
              ├─ llm.py         Kimi K2 via OpenRouter, JSON output, temperature 0.2
              └─ prompts.py     system prompt + JSON contract + validation
              ▼
         /api/tts (Kokoro, local ONNX)  ──> audio
```

## Run it on Replit

1. Create a Python Repl and upload these files (keep the folder structure).
2. **Secrets** (padlock icon): add
   - `OPENROUTER_API_KEY` = your key
   - `TEXT_MODEL` = `moonshotai/kimi-k2-0905` (optional, this is the default)
   - `VISION_MODEL` = any vision-capable model slug from openrouter.ai/models (default: `google/gemini-2.5-flash`; check it is still listed)
3. Install: `pip install -r requirements.txt`
4. Download the Kokoro model files into `data/` (about 300 MB, one time):
   ```bash
   mkdir -p data && cd data
   curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
   curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
   cd ..
   ```
   If the URLs have moved, see the kokoro-onnx GitHub repo. If Replit storage or RAM is tight, skip this: the page falls back to the browser's built-in voice automatically.
5. Cache the labels for your demo drugs (needs internet once):
   ```bash
   python prefetch.py ibuprofen acetaminophen amoxicillin metformin
   ```
   openFDA uses US names (acetaminophen = paracetamol). After this the demo works even if api.fda.gov is slow.
6. Run: `uvicorn main:app --host 0.0.0.0 --port 8000`  (set this as the Repl's run command)
7. Test: open the Repl URL, or `curl localhost:8000/api/health`.

## How the retrieval works

1. The confirmed drug name goes to `fetch_label()`, which queries openFDA by generic, brand or substance name and caches the best record in `data/cache/`.
2. `label_to_chunks()` splits the label sections into ~900-character chunks.
3. `build_context()` always includes the core sections (uses, dosing, warnings, interactions, pregnancy) and, if the person typed a question, adds the best-matching extra chunks using a small BM25-style scorer.
4. If nothing reliable is found, `/api/explain` returns a safe "ask a pharmacist" answer and **never calls the LLM**.
5. The patient profile is **not** retrieved: it is a small structured form, injected straight into the prompt and checked in code by `safety.py`.

To scale up: index more sources (DailyMed, WHO Essential Medicines, Jordan FDA directory) into Postgres + pgvector with a multilingual embedding model (bge-m3), and replace `rank_chunks` with a vector search over the same chunk format.

## Before real users

- Have a pharmacist or clinician review outputs for your target drugs.
- Replace the string-matching interaction check with a real interaction service (e.g. RxNav).
- Get a privacy review: never send names or IDs to the LLM, keep `provider.data_collection = deny`, and write a retention policy.
- Add Arabic and other languages: LLM output in the user's language plus Piper or MMS-TTS for audio.
