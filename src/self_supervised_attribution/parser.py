import io
from typing import List, Optional

import fitz
import pytesseract
from PIL import Image

from .schemas import OCR, Box, Document, Page


def extract_document_text(
    path: str, max_pages: Optional[int] = None, verbose: bool = False
) -> Optional[List[Page]]:
    """
    Extracts text from a pdf document.
    """
    assert path.endswith(".pdf")

    doc = fitz.open(path)

    d = Document()

    if (max_pages is not None) and (len(doc) > max_pages):
        return None

    for page_num in range(len(doc)):
        if verbose:
            print(f"Page {page_num + 1}/{len(doc)}")

        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n_boxes = len(ocr_data["text"])

        p = Page(number=page_num)

        for i in range(n_boxes):
            if int(ocr_data["conf"][i]) > 0:
                text = ocr_data["text"][i]
                x, y, w, h = (
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["width"][i],
                    ocr_data["height"][i],
                )

                p.ocr.append(OCR(text=text, box=Box(x=x, y=y, w=w, h=h)))

        d.pages.append(p)

    return d


def linearize_page(ocr_list: List[OCR]) -> str:
    """
    Linearizes a list of OCR objects into a string, preserving layout structure.
    """

    # ---- process lines ----

    lines = list()

    l_text = ""
    last_y = [0, 0]
    for _el in ocr_list:
        new_y = [_el.box.y, _el.box.y + _el.box.h]
        y_c = (new_y[0] + new_y[1]) / 2

        if (y_c < last_y[0]) or (last_y[1] < y_c):
            lines.append(l_text.strip())
            l_text = ""
            last_y = new_y

        if l_text.endswith("-") and _el.text == "":
            continue
        if l_text.endswith(" ") and _el.text == "":
            continue

        l_text += " " + _el.text

    if l_text:
        lines.append(l_text)

    # ---- merge ----

    page_text = ""
    for line in lines:
        page_text += "\n" + line.strip()

    return page_text.strip()
