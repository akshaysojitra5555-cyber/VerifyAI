import io
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)


def ocr_pdf(file_bytes: bytes) -> str:
    document = fitz.open(stream=file_bytes, filetype="pdf")

    all_text = []

    for page in document:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = Image.open(io.BytesIO(pix.tobytes("png")))

        text = ocr_image(image)
        all_text.append(text)

    document.close()

    return "\n\n".join(all_text)


def extract_text(filename: str, file_bytes: bytes) -> str:
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return ocr_pdf(file_bytes)

    if extension in [".jpg", ".jpeg", ".png"]:
        image = Image.open(io.BytesIO(file_bytes))
        return ocr_image(image)

    raise ValueError("Unsupported file format")