from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    VlmEngineType,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


class DoclingPdfConverter:
    def __init__(self, port: int, served_model_name: str):
        self.port = port
        self.base_url = f"http://localhost:{port}"

        self.prepare_converter(served_model_name)

    def prepare_converter(self, served_model_name: str):
        vlm_options = VlmConvertOptions.from_preset(
            "granite_docling",
            engine_options=ApiVlmEngineOptions(
                runtime_type=VlmEngineType.API,  # Generic API type
                url=f"{self.base_url}/v1/chat/completions",
                params={
                    "model": served_model_name,
                    "max_tokens": 4096,
                    "skip_special_tokens": False,
                    "temperature": 0.0,
                    "extra_body": {
                        "prompt": "Convert this page to docling",
                        "format": "doctags",
                    },
                },
                timeout=90,
            ),
        )

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True,
        )

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                )
            }
        )

    def __call__(self, pdf_path: str) -> str:
        r = self.doc_converter.convert(pdf_path)
        mk = r.document.export_to_markdown(image_placeholder="")

        return mk
