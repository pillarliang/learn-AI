from typing import Optional
from translator.pdf_parser import PDFParser
from translator.translation_chain import TranslationChain
from translator.writer import Writer


class PDFTranslator:
    def __init__(self, model_name: str):
        self.pdf_parser = PDFParser()
        self.translate_chain = TranslationChain(model_name)
        self.writer = Writer()

    def translate_pdf(
        self,
        input_file: str,
        output_file_format: str,
        source_language: str,
        target_language: str,
        pages: Optional[int] = None,
    ):
        # 解析
        self.book = self.pdf_parser.parse_pdf(input_file, pages)
        # 翻译
        for page_idx, page in enumerate(self.book.pages):
            for content_idx, content in enumerate(page.contents):
                translation, status = self.translate_chain.run(
                    content, source_language, target_language
                )
                self.book.pages[page_idx].contents[content_idx].set_translation(
                    translation, status
                )

        # 导出
        return self.writer.save_translated_book(self.book, output_file_format)
