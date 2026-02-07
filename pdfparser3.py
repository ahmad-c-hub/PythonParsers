import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional

import fitz
import numpy as np
import cv2
import pytesseract

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

WS_RE = re.compile(r"\s+")
COLON_RE = re.compile(r"^(.{2,80}?):\s*(.{0,200})$")

def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())

def rect_to_list(r: fitz.Rect) -> List[float]:
    return [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]

def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def render_page_rgb(doc: fitz.Document, page_index: int, dpi: int = 300) -> np.ndarray:
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return img

def preprocess_for_ocr(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return thr

def words_on_page(page: fitz.Page) -> List[Tuple[float, float, float, float, str]]:
    words = page.get_text("words") or []
    out = []
    for w in words:
        x0, y0, x1, y1, text, *_ = w
        t = (text or "").strip()
        if t:
            out.append((x0, y0, x1, y1, t))
    return out

def infer_label_from_words(words, rect: fitz.Rect) -> str:
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    h = y1 - y0
    w = x1 - x0

    left = []
    for wx0, wy0, wx1, wy1, t in words:
        if wx1 <= x0 - 5 and wx0 >= max(0.0, x0 - 300):
            if overlap_1d(wy0, wy1, y0 - 0.35*h, y1 + 0.35*h) > 0:
                left.append((x0 - wx1, wy0, t))
    left.sort(key=lambda z: (z[0], z[1]))
    if left:
        return norm_ws(" ".join(t for _, _, t in left[:10]))[:120]

    above = []
    for wx0, wy0, wx1, wy1, t in words:
        if wy1 <= y0 - 3 and wy0 >= max(0.0, y0 - 140):
            if overlap_1d(wx0, wx1, x0 - 0.2*w, x1 + 0.2*w) > 0:
                above.append((y0 - wy1, wx0, t))
    above.sort(key=lambda z: (z[0], z[1]))
    if above:
        return norm_ws(" ".join(t for _, _, t in above[:12]))[:120]

    return ""

def extract_ordered_widgets(doc: fitz.Document) -> List[Dict[str, Any]]:
    out = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        words = words_on_page(page)
        for w in list(page.widgets() or []):
            out.append({
                "page": pno + 1,
                "label": infer_label_from_words(words, w.rect),
                "value": getattr(w, "field_value", None),
                "rect": rect_to_list(w.rect),
                "source": "acroform",
                "field_name": getattr(w, "field_name", "") or "",
                "confidence": 0.95
            })
    out.sort(key=lambda d: (d["page"], d["rect"][1], d["rect"][0]))
    return out

def extract_text_kv(doc: fitz.Document) -> List[Dict[str, Any]]:
    out = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        lines = [norm_ws(l) for l in (page.get_text("text") or "").splitlines() if norm_ws(l)]
        for i, line in enumerate(lines):
            m = COLON_RE.match(line)
            if m:
                out.append({
                    "page": pno + 1,
                    "label": norm_ws(m.group(1)),
                    "value": norm_ws(m.group(2)),
                    "rect": [0.0, float(i), 0.0, float(i)],
                    "source": "text_kv",
                    "field_name": None,
                    "confidence": 0.65
                })
        for i in range(len(lines) - 1):
            if lines[i].endswith(":") and not lines[i + 1].endswith(":"):
                out.append({
                    "page": pno + 1,
                    "label": norm_ws(lines[i][:-1]),
                    "value": norm_ws(lines[i + 1]),
                    "rect": [0.0, float(i) + 0.5, 0.0, float(i) + 0.5],
                    "source": "text_kv",
                    "field_name": None,
                    "confidence": 0.55
                })
    out.sort(key=lambda d: (d["page"], d["rect"][1], d["rect"][0]))
    return out

def extract_ocr_kv(doc: fitz.Document, dpi: int = 300) -> List[Dict[str, Any]]:
    out = []
    for pno in range(doc.page_count):
        img_rgb = render_page_rgb(doc, pno, dpi)
        img_bin = preprocess_for_ocr(img_rgb)
        data = pytesseract.image_to_data(
            img_bin,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6"
        )
        words = []
        for i in range(len(data["text"])):
            txt = norm_ws(data["text"][i])
            if not txt:
                continue
            conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
            if conf < 35:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append((x, y, x + w, y + h, txt, conf))

        words.sort(key=lambda t: (t[1], t[0]))
        lines = []
        for w in words:
            if not lines or abs(w[1] - lines[-1][0][1]) > 10:
                lines.append([w])
            else:
                lines[-1].append(w)

        for line in lines:
            line.sort(key=lambda t: t[0])
            text = norm_ws(" ".join(t[4] for t in line))
            if not text:
                continue
            m = COLON_RE.match(text)
            if not m:
                continue
            x0 = min(t[0] for t in line)
            y0 = min(t[1] for t in line)
            x1 = max(t[2] for t in line)
            y1 = max(t[3] for t in line)
            avg_conf = sum(t[5] for t in line) / len(line)
            out.append({
                "page": pno + 1,
                "label": norm_ws(m.group(1)),
                "value": norm_ws(m.group(2)),
                "rect": [float(x0), float(y0), float(x1), float(y1)],
                "source": "ocr_kv",
                "field_name": None,
                "confidence": max(0.4, min(0.8, avg_conf / 100.0))
            })
    out.sort(key=lambda d: (d["page"], d["rect"][1], d["rect"][0]))
    return out

def parse_any_pdf(pdf_path: str, dpi: int = 300) -> Dict[str, Any]:
    if not os.path.exists(pdf_path):
        return {"errors": ["file not found"]}

    doc = fitz.open(pdf_path)
    try:
        widgets = extract_ordered_widgets(doc)
        if widgets:
            ordered = widgets
            doc_type = "form_widgets"
            preferred = "acroform"
        else:
            text_kv = extract_text_kv(doc)
            if text_kv:
                ordered = text_kv
                doc_type = "text_pdf"
                preferred = "text_kv"
            else:
                ordered = extract_ocr_kv(doc, dpi)
                doc_type = "scanned_pdf"
                preferred = "ocr_kv"

        return {
            "meta": {
                "document": os.path.basename(pdf_path),
                "pages": doc.page_count,
                "dpi": dpi,
                "doc_type": doc_type,
                "preferred_extraction": preferred,
                "items": len(ordered)
            },
            "ordered_fields": ordered,
            "errors": []
        }
    finally:
        doc.close()

if __name__ == "__main__":
    data = parse_any_pdf("testing_redacted_document.pdf", dpi=300)
    with open("parsed_any_pdf.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Saved -> parsed_any_pdf.json")
