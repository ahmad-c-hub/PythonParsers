import json
import re
from typing import Any, Dict, List, Tuple, Optional
import os

import fitz  # PyMuPDF

# Optional OCR fallback (only used if option text isn't extractable)
OCR_ENABLED = False
try:
    import pytesseract
    import numpy as np
    import cv2
    OCR_ENABLED = True
except Exception:
    OCR_ENABLED = False


import pytesseract

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    print("[warning] Tesseract exe not found at expected path.")


WS_RE = re.compile(r"\s+")
CAMEL_RE = re.compile(r"([a-z])([A-Z])")
SUFFIX_HASH_RE = re.compile(r"(\[[0-9]+\])([a-z0-9]{8,})$", re.IGNORECASE)

DROP_TOKENS = (
    "reset", "submit", "print", "button", "btn", "pushbutton",
    "signature", "sig", "barcode", "qr", "attachment", "save", "clear"
)

NULL_STRINGS = {"", "n/a", "na", "none", "null", "not_provided", "not provided", "n.a."}

# IMPORTANT: Do NOT map plain "1" to True globally, it breaks numeric fields.
CODE_MAP = {
    "/Off": False, "Off": False,
    "/On": True,  "On": True,
    "/Yes": True, "Yes": True,
    "/No": False, "No": False,
}


def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())


def simplify_field_name(name: str) -> str:
    if not name:
        return ""
    n = name.strip()
    n = SUFFIX_HASH_RE.sub(r"\1", n)
    return n


def should_drop(s: str) -> bool:
    s = (s or "").lower()
    return any(t in s for t in DROP_TOKENS)


def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = norm_ws(v)
        if s in CODE_MAP:
            return CODE_MAP[s]
        if s.lower() in {x.lower() for x in NULL_STRINGS}:
            return None
        return s
    if isinstance(v, list):
        out = []
        for x in v:
            nx = normalize_value(x)
            if nx is not None:
                out.append(nx)
        return out if out else None
    if isinstance(v, (int, float, bool)):
        return v
    return norm_ws(str(v))


def is_nullish(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return norm_ws(v).lower() in {x.lower() for x in NULL_STRINGS}
    if isinstance(v, list):
        return all(is_nullish(x) for x in v)
    return False


def semantic_key_from_field_name(field_name: str) -> str:
    if not field_name:
        return ""

    parts = field_name.split(".")
    last = parts[-1]

    last = re.sub(r"\[[0-9]+\]", "", last)
    last = re.sub(r"^(pt|p|part)?\d+\s*", "", last, flags=re.IGNORECASE)
    last = re.sub(r"^(line|pt|p|part)\d+[a-z]*_", "", last, flags=re.IGNORECASE)
    last = re.sub(r"^(pt|p|part)\d+_", "", last, flags=re.IGNORECASE)
    last = re.sub(r"^[A-Za-z]+_", "", last)

    last = last.replace("_", " ")
    last = CAMEL_RE.sub(r"\1 \2", last)
    last = norm_ws(last)

    return last.title() if last else ""


# -----------------------------
# Checkbox/radio detection (REAL FIX)
# -----------------------------

def rect_wh(rect: List[float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = rect
    return abs(x1 - x0), abs(y1 - y0)


def looks_like_choice_widget(item: Dict[str, Any]) -> bool:
    """
    Detect checkbox/radio by:
    - acroform source
    - rect exists and is small (typical checkboxes are ~8-16pt square)
    - value looks like Off/On/Yes/No OR export values A/B/C... OR empty
    """
    if (item.get("source") or "").lower() != "acroform":
        return False

    rect = item.get("rect")
    if not (isinstance(rect, list) and len(rect) == 4):
        return False

    w, h = rect_wh(rect)
    # typical checkbox square ~10x10, allow generous bounds
    if not (5 <= w <= 25 and 5 <= h <= 25):
        return False

    raw = item.get("value")
    if raw is None:
        return True

    s = norm_ws(str(raw))
    # Very common checkbox/radio states or export values
    if s in ("", "Off", "/Off", "On", "/On", "Yes", "/Yes", "No", "/No", "1", "/1"):
        return True
    # Export values like A/B/C or other short tokens
    if len(s) <= 3 and s.isalnum():
        return True

    return False


def is_selected_choice(item: Dict[str, Any]) -> bool:
    """
    Selected if it's not Off/empty/false-ish.
    """
    raw = item.get("value")
    if raw is None:
        return False

    if isinstance(raw, bool):
        return raw

    s = norm_ws(str(raw))
    if not s:
        return False
    if s in ("Off", "/Off", "0", "/0", "False", "false", "No", "/No"):
        return False
    return True


# -----------------------------
# PDF layout extraction
# -----------------------------

Word = Tuple[float, float, float, float, str]  # x0,y0,x1,y1,word
Line = Tuple[float, float, str, List[Word]]    # y_center, x0_min, text, words


def get_page_words(doc: fitz.Document, page_index_0: int) -> List[Word]:
    page = doc.load_page(page_index_0)
    words = page.get_text("words")
    out: List[Word] = []
    for w in words:
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        t = norm_ws(txt)
        if t:
            out.append((x0, y0, x1, y1, t))
    return out


def group_words_into_lines(words: List[Word], y_tol: float = 2.5) -> List[Line]:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (((w[1] + w[3]) / 2.0), w[0]))

    lines: List[List[Word]] = []
    cur: List[Word] = []
    cur_y: Optional[float] = None

    for w in words_sorted:
        y = (w[1] + w[3]) / 2.0
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur.append(w)
            cur_y = y if cur_y is None else (cur_y * 0.7 + y * 0.3)
        else:
            lines.append(cur)
            cur = [w]
            cur_y = y

    if cur:
        lines.append(cur)

    out: List[Line] = []
    for ln in lines:
        ln_sorted = sorted(ln, key=lambda w: w[0])
        y_center = sum((w[1] + w[3]) / 2.0 for w in ln_sorted) / len(ln_sorted)
        x0_min = min(w[0] for w in ln_sorted)
        text = norm_ws(" ".join(w[4] for w in ln_sorted))
        out.append((y_center, x0_min, text, ln_sorted))
    return out


def find_line_near_y(lines: List[Line], y: float) -> Tuple[int, float]:
    best_i = -1
    best_d = 1e9
    for i, (ly, _, _, _) in enumerate(lines):
        d = abs(ly - y)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def extract_option_text(lines: List[Line], rect: List[float], right_pad: float = 3.0, y_tol: float = 6.0) -> str:
    x0, y0, x1, y1 = rect
    y_center = (y0 + y1) / 2.0

    idx, dy = find_line_near_y(lines, y_center)
    if idx < 0 or dy > y_tol:
        return ""

    _, _, _, lwords = lines[idx]
    right_words = [w for w in lwords if w[0] >= (x1 + right_pad)]
    right_words = sorted(right_words, key=lambda w: w[0])
    return norm_ws(" ".join(w[4] for w in right_words))


def ocr_option_text(doc: fitz.Document, page_index_0: int, rect: List[float]) -> str:
    """
    OCR a region to the right of the checkbox (fallback only).
    """
    if not OCR_ENABLED:
        return ""

    page = doc.load_page(page_index_0)
    x0, y0, x1, y1 = rect

    # region to right of checkbox (wide strip)
    clip = fitz.Rect(x1 + 2, y0 - 2, x1 + 320, y1 + 12)
    pix = page.get_pixmap(clip=clip, dpi=300)

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(gray, config="--psm 6")
    return norm_ws(text)


# -----------------------------
# Main cleaning
# -----------------------------

def clean_parsed_json(pdf_path: str, input_path: str, output_path: str, max_items: int = 5000) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    items = (data.get("ordered_fields") or [])[:max_items]

    # Debug reality
    choice_candidates = [it for it in items if looks_like_choice_widget(it)]
    print(f"[debug] total_items={len(items)} choice_candidates={len(choice_candidates)} ocr_enabled={OCR_ENABLED}")

    doc = fitz.open(pdf_path)
    line_cache: Dict[int, List[Line]] = {}

    llm_fields: List[Dict[str, Any]] = []

    # 1) Group choice widgets by (page, label) — THIS matches your data
    choice_groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    normal_items: List[Dict[str, Any]] = []

    for it in items:
        if looks_like_choice_widget(it):
            p = it.get("page")
            lbl = norm_ws(it.get("label") or "")
            if isinstance(p, int) and p > 0 and lbl:
                choice_groups.setdefault((p, lbl), []).append(it)
            else:
                # if label missing, just treat as normal (we can't group sanely)
                normal_items.append(it)
        else:
            normal_items.append(it)

    # 2) Emit normal (non-choice) items
    for it in normal_items:
        v = normalize_value(it.get("value"))
        if is_nullish(v):
            continue

        fn = simplify_field_name(it.get("field_name") or "")
        lbl = norm_ws(it.get("label") or "")

        if should_drop(fn) or should_drop(lbl):
            continue

        key = semantic_key_from_field_name(fn)
        if not key:
            if lbl and not should_drop(lbl):
                key = lbl[:120]
            else:
                continue

        llm_fields.append({
            "page": it.get("page"),
            "key": key,
            "value": v,
            "source": it.get("source")
        })

    # 3) Emit grouped MCQ/checkbox questions
    for (page_1, question_label), group_items in choice_groups.items():
        selected = [it for it in group_items if is_selected_choice(it)]
        if not selected:
            continue

        p0 = page_1 - 1
        if p0 not in line_cache:
            words = get_page_words(doc, p0)
            line_cache[p0] = group_words_into_lines(words)

        lines = line_cache[p0]

        answers: List[str] = []
        for it in selected:
            rect = it.get("rect")
            if not (isinstance(rect, list) and len(rect) == 4):
                continue

            opt = extract_option_text(lines, rect)
            if not opt:
                # OCR fallback if PDF text isn't extractable
                opt = ocr_option_text(doc, p0, rect)

            if not opt:
                # last fallback: field_name semantic (better than A/B)
                fn = simplify_field_name(it.get("field_name") or "")
                opt = semantic_key_from_field_name(fn) or norm_ws(str(it.get("value") or ""))

            if opt:
                answers.append(opt)

        # De-dupe answers
        seen = set()
        deduped = []
        for a in answers:
            a2 = norm_ws(a)
            if a2 and a2 not in seen:
                seen.add(a2)
                deduped.append(a2)

        if not deduped:
            continue

        llm_fields.append({
            "page": page_1,
            "key": question_label[:120],
            "value": deduped[0] if len(deduped) == 1 else deduped,
            "source": "acroform"
        })

    doc.close()

    out = {
        "meta": {
            "document": meta.get("document"),
            "pages": meta.get("pages"),
            "dpi": meta.get("dpi"),
            "doc_type": meta.get("doc_type"),
            "preferred_extraction": meta.get("preferred_extraction"),
            "items_in": meta.get("items"),
            "items_out": len(llm_fields),
        },
        "llm_fields": llm_fields,
        "llm_rules": [
            "Use llm_fields as structured data.",
            "Do not invent missing values.",
            "If duplicate keys exist, prefer non-empty values and earlier pages."
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    clean_parsed_json(
        pdf_path="testing_redacted_document.pdf",
        input_path="parsed_any_pdf1.json",
        output_path="parsed_any_pdf1.llm.json"
    )
