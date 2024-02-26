import os
import sys
from translator import PDFTranslator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # instantiate a class
    pdf_translator = PDFTranslator(model_name="gpt-3.5-turbo")
    pdf_translator.translate_pdf(
        f"{CURRENT_DIR}/test.pdf",
        "pdf",
        target_language="Chinese",
        source_language="English",
    )
