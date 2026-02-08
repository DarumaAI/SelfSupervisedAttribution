import glob
import os
from typing import List

import arxiv
import requests
from tqdm.auto import tqdm

from self_supervised_attribution.parser import extract_document_text, linearize_page


def automatic_loader(queries: List[str], max_results: int):
    results = set()
    client = arxiv.Client()
    for q in queries:
        search = arxiv.Search(
            query=q,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        for r in client.results(search):
            results.add(r.pdf_url)

    pdf_dir = "./pdf"
    os.makedirs(pdf_dir, exist_ok=True)
    for url in tqdm(results, desc="Downloading papers"):
        name = os.path.basename(url).replace(".pdf", "")
        response = requests.get(url)
        pdf_path = f"{pdf_dir}/{name}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)

    txt_dir = "./txt"
    os.makedirs(txt_dir, exist_ok=True)
    for pdf_path in tqdm(glob.glob(pdf_dir), desc="Extracting text"):
        doc = extract_document_text(pdf_path)
        content = "\n\n".join([linearize_page(page.ocr) for page in doc.pages])
        txt_path = f"{txt_dir}/{os.path.basename(pdf_path).replace('.pdf', '.txt')}"
        with open(txt_path, "w") as f:
            f.write(content)
