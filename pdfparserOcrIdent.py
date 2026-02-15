import os
import io
import re
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

def extract_native_blocks(page: fitz.Page):

    blocks = page.get_text("blocks") 
    out = []
    for b in blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        text = clean_text(text)
        if text:
            out.append((y0, x0, text))
    out.sort(key=lambda t: (t[0], t[1]))
    return out

def extract_page_native_text(page: fitz.Page) -> str:
    return clean_text(page.get_text("text"))

def extract_ocr_text(page: fitz.Page, dpi: int = 300) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return clean_text(pytesseract.image_to_string(img))

def pdf_to_markdown_stream_tagged(
    pdf_path: str,
    output_md_path: str,
    dpi: int = 300,
    ocr_trigger_min_chars: int = 40,
    ocr_trigger_min_words: int = 8
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    md = [f"# {os.path.basename(pdf_path)}\n"]

    for page_idx, page in enumerate(doc, start=1):
        md.append(f"## Page {page_idx}\n")

        native_full = extract_page_native_text(page)
        native_words = len(native_full.split())
        needs_ocr = (len(native_full) < ocr_trigger_min_chars) or (native_words < ocr_trigger_min_words)

        if needs_ocr:
            ocr_text = extract_ocr_text(page, dpi=dpi)
            if ocr_text:
                md.append(f"**[OCR]**\n\n{ocr_text}\n")
            else:
                md.append("**[OCR]**\n\n_(no text extracted)_\n")
        else:
            blocks = extract_native_blocks(page)
            if blocks:
                for _, __, txt in blocks:
                    md.append(f"**[NATIVE]**\n\n{txt}\n")
            else:
                md.append("**[NATIVE]**\n\n_(no text extracted)_\n")

        md.append("\n---\n")

    doc.close()

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md).strip() + "\n")

    print(f"Saved: {output_md_path}")

if __name__ == "__main__":

    pdf_to_markdown_stream_tagged(
        pdf_path="testing_redacted_document.pdf",
        output_md_path="outputOCR.md",
        dpi=300
    )
