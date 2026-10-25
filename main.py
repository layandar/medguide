"""MedGuide backend (FastAPI).

Flow:  photo -> /api/identify -> user confirms name -> /api/explain -> /api/tts
"""

import base64
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import llm
import tts
from prompts import SYSTEM_PROMPT, VISION_PROMPT, build_user_message, validate_output
from retrieval import get_label_context
from safety import safety_flags

log = logging.getLogger("medguide")
app = FastAPI(title="MedGuide")

BASE = Path(__file__).parent
MAX_IMAGE_BYTES = 6 * 1024 * 1024

SAFE_FALLBACK = {
    "drug_name": "",
    "used_for": "",
    "how_to_take": [],
    "dose_and_timing": "Ask your pharmacist or doctor for your dose.",
    "with_food": "Ask your pharmacist.",
    "warnings": ["I could not find reliable information for this medicine."],
    "missed_dose": "Ask your pharmacist what to do if you miss a dose.",
    "see_a_doctor_if": [],
    "ask_pharmacist": True,
    "speech_text": (
        "I could not find reliable information about this medicine. "
        "Please show the box to a pharmacist or doctor before you take it."
    ),
    "sources": [],
}


# ----------------------------------------------------------------------------
# 1) Identify the medicine from a photo of the box/label
# ----------------------------------------------------------------------------
@app.post("/api/identify")
async def identify(photo: UploadFile = File(...)):
    data = await photo.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 6 MB).")
    mime = photo.content_type or "image/jpeg"
    if not mime.startswith("image/"):
        raise HTTPException(400, "Please upload an image.")
    try:
        result = llm.identify_from_image(base64.b64encode(data).decode(), mime, VISION_PROMPT)
    except (httpx.HTTPError, RuntimeError, ValueError) as e:
        log.warning("identify failed: %s", e)
        raise HTTPException(502, "Could not read the image. Please type the medicine name.")

    # Prefer the active ingredient for lookup; keep the brand for display.
    lookup = result.get("generic_name") or result.get("brand_name")
    return {
        "lookup_name": lookup,
        "brand_name": result.get("brand_name"),
        "generic_name": result.get("generic_name"),
        "strength": result.get("strength"),
        "form": result.get("form"),
        "confidence": result.get("confidence", "low"),
    }


# ----------------------------------------------------------------------------
# 2) Explain: retrieval -> deterministic checks -> LLM -> validated JSON
# ----------------------------------------------------------------------------
class Patient(BaseModel):
    age_group: str = Field("adult", pattern="^(infant|child|adult|older_adult)$")
    pregnant_or_breastfeeding: bool = False
    allergies: list[str] = []
    current_medications: list[str] = []
    language: str = "en"


class ExplainRequest(BaseModel):
    drug_name: str = Field(..., min_length=2, max_length=80)  # user-confirmed name
    question: str | None = Field(None, max_length=300)
    patient: Patient = Patient()


@app.post("/api/explain")
def explain(req: ExplainRequest):
    # --- RAG step: ground everything in the official label ---
    found = get_label_context(req.drug_name, req.question)
    if not found:
        # No reliable source -> do NOT let the model answer from memory.
        return {**SAFE_FALLBACK, "drug_name": req.drug_name}

    patient = req.patient.model_dump()
    flags = safety_flags(found["label"], patient)

    # Send the model only what it needs. No name, no ID, no free text beyond the question.
    patient_for_llm = {
        "age_group": patient["age_group"],
        "pregnant_or_breastfeeding": patient["pregnant_or_breastfeeding"],
        "allergies": patient["allergies"],
        "other_medicines": patient["current_medications"],
        "reply_language": patient["language"],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(req.drug_name, patient_for_llm, found["context"], flags, req.question)},
    ]

    for attempt in range(2):  # one retry if the JSON is malformed
        try:
            out = llm.chat_json(messages, llm.TEXT_MODEL)
        except (httpx.HTTPError, RuntimeError, ValueError) as e:
            log.warning("llm failed (attempt %s): %s", attempt + 1, e)
            continue
        problems = validate_output(out)
        if not problems:
            # Code-level guarantee: flags always force the pharmacist notice.
            if flags:
                out["ask_pharmacist"] = True
            out["sources"] = [s for s in out.get("sources", []) if s in found["sections"]] or found["sections"]
            out["safety_flags"] = flags
            return out
        log.warning("invalid model output: %s", problems)

    return {**SAFE_FALLBACK, "drug_name": req.drug_name}


# ----------------------------------------------------------------------------
# 3) Text to speech
# ----------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1200)
    lang: str = "en"


@app.post("/api/tts")
def speak(req: TTSRequest):
    if not tts.available():
        raise HTTPException(503, "TTS model files missing; the browser voice will be used instead.")
    try:
        wav = tts.synthesize_wav(req.text, req.lang)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(content=wav, media_type="audio/wav")


@app.get("/api/health")
def health():
    return {"ok": True, "tts_ready": tts.available(), "text_model": llm.TEXT_MODEL, "vision_model": llm.VISION_MODEL}


# ----------------------------------------------------------------------------
# Frontend
# ----------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
