import json
import logging
import time

from pathlib import Path
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def pdf_parser(input_doc_path):
    """
    Parses a PDF document and exports its content in multiple formats.

    This function uses Docling processing pipeline with OCR and table structure recognition
    to convert the given PDF file into structured data. It saves the output in JSON, 
    plain text, Markdown, and custom document tags formats into a subdirectory named 
    after the input file.

    Parameters:
        input_doc_path (str or Path): Path to the input PDF document.

    Outputs:
        Creates a directory in `parsed_reports/` containing:
            - `<filename>.json`       : Structured JSON representation.
            - `<filename>.txt`        : Plain text export of the document.
            - `<filename>.md`         : Markdown export of the document.
            - `<filename>.doctags`    : Tokenized document tags.
    """
    logging.basicConfig(level=logging.INFO) 
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options.lang = ["en"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
  

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    end_time = time.time() - start_time

    logging.info(f"Document converted in {end_time:.2f} seconds.")

    ## Export results
    filename = str(input_doc_path).split('\\')[1].rstrip('.pdf')
    output_dir = Path(f"data//parsed_docs//{filename}")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_result.input.file.stem


    # JSON format:
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_result.document.export_to_dict()))

    # Text format:
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_text())

    # Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())

    # Document Tags format:
    with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_document_tokens())


####    Todo: Create a wrapper parsing all available reports    ####
