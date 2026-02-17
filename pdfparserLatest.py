import os
import re
import io
import fitz
import pytesseract
from PIL import Image

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

WS_RE = re.compile(r"[ \t]+")

def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = WS_RE.sub(" ", s)
    return s.strip()

def weak_text(s: str, min_chars: int, min_words: int) -> bool:
    s = clean_text(s)
    return len(s) < min_chars or len(s.split()) < min_words

def ocr_page(page: fitz.Page, dpi: int = 300) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return clean_text(pytesseract.image_to_string(img))

def md_from_single_page(doc: fitz.Document, page_index: int) -> str:
    import pymupdf4llm
    one = fitz.open()
    one.insert_pdf(doc, from_page=page_index, to_page=page_index)
    md = pymupdf4llm.to_markdown(one)
    one.close()
    return (md or "").strip()

def extract_page_form_fields(page: fitz.Page, y_tol: float = 3.0):
    widgets = page.widgets()
    if not widgets:
        return []

    items = []
    for w in widgets:
        name = (w.field_name or "").strip()
        if not name:
            continue

        value = w.field_value
        value = "" if value is None else str(value).strip()

        ftype = getattr(w, "field_type", None)
        ftype = "" if ftype is None else str(ftype).strip()

        r = getattr(w, "rect", None)
        if r is None:
            x0 = y0 = 0.0
        else:
            x0 = float(r.x0)
            y0 = float(r.y0)

        items.append({
            "name": name,
            "value": value,
            "type": ftype,
            "x0": x0,
            "y0": y0,
        })

    def sort_key(it):
        y_bucket = round(it["y0"] / y_tol)
        return (y_bucket, it["x0"], it["name"])

    items.sort(key=sort_key)

    seen = set()
    out = []
    for it in items:
        k = (it["name"], it["value"])
        if k in seen:
            continue
        seen.add(k)
        out.append({"name": it["name"], "value": it["value"], "type": it["type"]})

    return out

def render_fields_md(fields):
    lines = ["### Form Fields (AcroForm)\n"]
    for f in fields:
        val = f["value"] if f["value"] else "_(empty)_"
        lines.append(f"- **{f['name']}**: {val}")
    return "\n".join(lines).strip()

def pdf_to_markdown_clean_with_forms(
    pdf_path: str,
    output_md_path: str,
    dpi: int = 300,
    min_md_chars: int = 80,
    min_md_words: int = 15,
    y_tol: float = 3.0,
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    out = [f"# {os.path.basename(pdf_path)}\n"]

    for i in range(doc.page_count):
        page_num = i + 1
        page = doc[i]

        out.append(f"\n---\n## Page {page_num}\n")

        fields = extract_page_form_fields(page, y_tol=y_tol)
        if fields:
            out.append(render_fields_md(fields))
            out.append("")

        page_md = md_from_single_page(doc, i)

        if not page_md or weak_text(page_md, min_md_chars, min_md_words):
            ocr_txt = ocr_page(page, dpi=dpi)
            out.append("### Content\n")
            out.append(ocr_txt if ocr_txt else "_(No text extracted)_")
            out.append("\n> Source: OCR\n")
        else:
            out.append("### Content\n")
            out.append(page_md)
            out.append("\n> Source: PyMuPDF4LLM\n")

    doc.close()

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out).strip() + "\n")

    print(f"Saved: {output_md_path}")

if __name__ == "__main__":
    pdf_to_markdown_clean_with_forms(
        pdf_path="testing_redacted_document.pdf",
        output_md_path="output_with_forms.md",
        dpi=300,
        y_tol=3.0,
    )
