"""Deterministic safety checks that run in code, NOT in the LLM.

These are intentionally simple string checks over the retrieved label text.
They can only add warnings; they never remove any. For production, replace the
interaction check with a real service (e.g. RxNav interaction API) and a
proper allergy/ingredient mapping.
"""

import re


def _text(label: dict, *fields: str) -> str:
    out = []
    for f in fields:
        v = label.get(f)
        if isinstance(v, list):
            out.extend(str(x) for x in v)
        elif v:
            out.append(str(v))
    return " ".join(out).lower()


def _norm_list(items) -> list[str]:
    return [re.sub(r"[^a-z0-9 ]+", "", str(i).lower()).strip() for i in (items or []) if str(i).strip()]


def safety_flags(label: dict, patient: dict) -> list[str]:
    flags: list[str] = []

    # 1) Allergy vs. what is in the product
    active = _text(label, "active_ingredient", "substance_name", "inactive_ingredient")
    for allergy in _norm_list(patient.get("allergies")):
        if allergy and allergy in active:
            flags.append(f"The person says they are allergic to '{allergy}', and this medicine appears to contain it.")

    # 2) Current medicines mentioned in the label's interaction text
    interactions = _text(label, "drug_interactions", "warnings", "ask_doctor", "do_not_use")
    for med in _norm_list(patient.get("current_medications")):
        if med and re.search(rf"\b{re.escape(med)}\b", interactions):
            flags.append(f"The label mentions '{med}' (which the person also takes) in its interaction or warning text.")

    # 3) Pregnancy / breastfeeding
    if patient.get("pregnant_or_breastfeeding"):
        flags.append("The person is pregnant or breastfeeding. Check the pregnancy information with a pharmacist before use.")

    # 4) Children
    if patient.get("age_group") in ("infant", "child"):
        flags.append("The medicine is for a child. Dose depends on weight and age; confirm the exact dose with a pharmacist or doctor.")

    # 5) Boxed warning present
    if label.get("boxed_warning"):
        flags.append("This label has a boxed (most serious) warning. Read it with a pharmacist.")

    return flags
