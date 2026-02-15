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

def ocr_page(page: fitz.Page, dpi: int = 300) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return clean_text(pytesseract.image_to_string(img))

def extract_md_with_pymupdf4llm(doc: fitz.Document) -> str:
    """
    Returns Markdown for the whole document using pymupdf4llm.
    """
    import pymupdf4llm
    return pymupdf4llm.to_markdown(doc)

def pdf_to_markdown_pymupdf4llm_with_ocr_fallback(
    pdf_path: str,
    output_md_path: str,
    dpi: int = 300,
    ocr_trigger_min_chars: int = 60,
    ocr_trigger_min_words: int = 10,
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    base_md = extract_md_with_pymupdf4llm(doc).strip()
    ocr_sections = []
    for i, page in enumerate(doc, start=1):
        native_text = clean_text(page.get_text("text"))
        if (len(native_text) < ocr_trigger_min_chars) or (len(native_text.split()) < ocr_trigger_min_words):
            ocr_text = ocr_page(page, dpi=dpi)
            if ocr_text:
                ocr_sections.append(f"\n\n---\n\n## OCR Fallback — Page {i}\n\n{ocr_text}\n")

    doc.close()

    final_md = base_md
    if ocr_sections:
        final_md += "\n\n---\n\n# OCR Fallback Pages\n" + "".join(ocr_sections)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(final_md.strip() + "\n")

    print(f"Saved: {output_md_path}")

if __name__ == "__main__":
    pdf_to_markdown_pymupdf4llm_with_ocr_fallback(
        pdf_path="testing_redacted_document.pdf",
        output_md_path="outputPyMu4llmMD.md",
        dpi=300,
    )

