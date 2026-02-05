import os
import re
import json
from typing import Any, Dict, Optional

import fitz
import cv2
import numpy as np
import pytesseract
from pypdf import PdfReader

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

VALUE_MAPS: Dict[str, Dict[Any, Any]] = {
    "Gender": {
        "/0": "Male",
        "/1": "Female",
        "/Male": "Male",
        "/Female": "Female",
    },
    "Married": {
        "/Yes": "Yes",
        "/Off": "No",
        True: "Yes",
        False: "No",
    },
}

DROP_KEYS = {"ResetButton", "SubmitButton", "PrintButton"}
CHECKBOX_OR_RADIO_KEYS = {"Gender", "Married"}


def _normalize_whitespace(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_clean_scalar(v: Any) -> Any:
    if v is None:
        return None

    if isinstance(v, (str, int, float, bool)):
        return v

    if isinstance(v, list):
        return [_to_clean_scalar(x) for x in v]

    return str(v)


def clean_form_fields(raw_fields: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}

    for key, raw_val in raw_fields.items():
        if key in DROP_KEYS:
            continue

        v = _to_clean_scalar(raw_val)

        if isinstance(v, str):
            v = _normalize_whitespace(v)

        if isinstance(v, list):
            v = [x for x in v if x is not None]
            v = [_normalize_whitespace(x) if isinstance(x, str) else x for x in v]

        if key in VALUE_MAPS:
            mapper = VALUE_MAPS[key]
            if isinstance(v, list):
                v = [mapper.get(x, x) for x in v]
            else:
                v = mapper.get(v, v)

        if v is None:
            v = "unchecked_or_not_selected" if key in CHECKBOX_OR_RADIO_KEYS else "not_provided"

        if key.lower() == "id":
            v = str(v)

        cleaned[key] = v

    return cleaned


def make_llm_payload(
    cleaned_fields: Dict[str, Any],
    source: str,
    doc_name: str,
    extracted_text: str = "",
    ocr_pages: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    return {
        "meta": {
            "document": doc_name,
            "source": source,
        },
        "data": cleaned_fields,
        "text": extracted_text,
        "ocr_pages": ocr_pages or {},
        "rules": {
            "llm_instruction": "Use ONLY the provided fields. Do not invent missing values."
        }
    }


def extract_acroform_fields(pdf_path: str) -> dict:
    reader = PdfReader(pdf_path)
    fields = reader.get_fields() or {}
    out = {}

    for name, field in fields.items():
        v = field.get("/V")
        if v is None:
            v = field.get("/AS")
        out[name] = str(v) if v is not None else None

    return out


def extract_text_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_chunks = []
    for page in doc:
        t = page.get_text("text").strip()
        if t:
            text_chunks.append(t)
    return "\n".join(text_chunks).strip()


def render_page_to_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return thr


def ocr_image(img: np.ndarray) -> str:
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(img, config=config)


def parse_form(pdf_path: str) -> dict:
    doc_name = os.path.basename(pdf_path)

    raw_fields = extract_acroform_fields(pdf_path)
    cleaned_fields = clean_form_fields(raw_fields) if raw_fields else {}

    source_type = None
    extracted_text = ""
    ocr_pages: Dict[int, str] = {}

    if raw_fields:
        source_type = "acroform"

    text = extract_text_pymupdf(pdf_path)
    if text:
        extracted_text = text
        if source_type is None:
            source_type = "text_pdf"
        return make_llm_payload(
            cleaned_fields=cleaned_fields,
            source=source_type,
            doc_name=doc_name,
            extracted_text=extracted_text,
            ocr_pages={}
        )

    if source_type is None:
        source_type = "scanned_or_flattened"

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        img = render_page_to_image(page, dpi=300)
        prep = preprocess_for_ocr(img)
        ocr_pages[i] = ocr_image(prep).strip()

    return make_llm_payload(
        cleaned_fields=cleaned_fields,
        source=source_type,
        doc_name=doc_name,
        extracted_text="",
        ocr_pages=ocr_pages
    )


if __name__ == "__main__":
    data = parse_form("Form1.pdf")

    print("TYPE:", data["meta"]["source"])

    if data["data"]:
        print("\nLLM-READY FIELDS:")
        for k, v in data["data"].items():
            print(f"{k}: {v}")

    if data["text"]:
        print("\nTEXT (first 1000 chars):")
        print(data["text"][:1000])

    if data["ocr_pages"]:
        print("\nOCR (first page, first 1000 chars):")
        print(data["ocr_pages"].get(1, "")[:1000])

    with open("parsed_output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\nSaved: parsed_output.json")
