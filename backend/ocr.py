
import io
import os
import shutil
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


# Find Tesseract automatically on the deployment server.
# TESSERACT_CMD can optionally override the detected path.
TESSERACT_PATH = (
    os.getenv("TESSERACT_CMD")
    or shutil.which("tesseract")
)

# Windows fallback for local development.
if not TESSERACT_PATH and os.name == "nt":
    windows_path = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if windows_path.exists():
        TESSERACT_PATH = str(windows_path)

# Configure Tesseract when a valid path is available.
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def ocr_image(image: Image.Image) -> str:
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image)


def ocr_pdf(file_bytes: bytes) -> str:
    """Extract text from every page of a PDF using OCR."""
    document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    try:
        all_text = []

        for page in document:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            image = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            text = ocr_image(image)
            all_text.append(text)

        return "\n\n".join(all_text)

    finally:
        document.close()


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract text from supported PDF and image files."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return ocr_pdf(file_bytes)

    if extension in [".jpg", ".jpeg", ".png"]:
        image = Image.open(io.BytesIO(file_bytes))

        try:
            return ocr_image(image)
        finally:
            image.close()

    raise ValueError("Unsupported file format")
