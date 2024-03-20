import pprint
from typing import List

import pandas as pd

from smartpy.utility import os_util
from tinyllm.rag.document.document import Document, DocumentTypes


class Loader:

    def __init__(self, file_path):
        self.file_path = file_path

    @classmethod
    def load(self) -> Document:
        pass


class ExcelLoader(Loader):

    def __init__(self, file_path, sheets=None):
        super().__init__(file_path)
        self.excel_file = pd.ExcelFile(file_path, engine='openpyxl')
        self.sheets = sheets or list(self.excel_file.sheet_names)

    def get_content(self) -> str:
        file_content = ""
        for sheet in self.sheets:
            title = sheet.upper()
            file_content += f"## {title}\n"
            file_content += pd.read_excel(self.file_path, sheet_name=sheet, engine='openpyxl').to_markdown()

    def get_screenshots(self) -> List[str]:
        pass


doc_parser_system_role = f"""
ROLE:
You are an Image document parser. You will be provided an image and/or text content from a single document. Your goal is to extract 
structured data (sections, fields, descriptions) from this document using the provided image/text. 

OUTPUT FORMAT:
Your output should be a JSON object that properly structures the extracted data. Make sure the section
and field names are semantically meaningful.
"""

class ImageLoader(Loader):

    def __init__(self, file_path):
        super().__init__(file_path)
        self.doc_parser = Agent(
            name="Document Parser",
            system_role=doc_parser_system_role,
        )


    def load(self) -> Document:
        return ""


class PDFFormLoader(Loader):

    def __init__(self, file_path):
        super().__init__(file_path)
        from PyPDF2 import PdfReader
        self.pdf_reader = PdfReader(open(file_path, "rb"))

    def load(self,
             parse_images=False) -> str:
        form_content = self.pdf_reader.get_form_text_fields()
        form_content = {str(k): (float(v) if v.isdigit() else v) for k, v in form_content.items() if
                        v is not None and len(v) > 0}
        form_content = pprint.pformat(form_content)
        pdf_screenshots = self.get_screeshots(self.file_path)
        image_loaders = [Document(metadata={'image_url': image_path}, type=DocumentTypes.IMAGE) for image_path in
                      pdf_screenshots]
        if parse_images:
            for img_doc in image_docs:
                img_doc.load()

        final_doc = Document(content=form_content,
                             metadata={'file_path': self.file_path},
                             images=image_docs)
        return final_doc

    def get_screeshots(self, pdf_path, dpi=500):
        from pdf2image import convert_from_path
        import os

        images_path = os_util.getTempDir('lendmarq-docs')
        pages = convert_from_path(pdf_path, dpi)
        image_paths = []
        for count, page in enumerate(pages):
            file_name = f'page_{count}.jpg'
            img_path = os.path.join(images_path, file_name)
            page.save(img_path, 'JPEG')
            image_paths.append(img_path)
        return image_paths


loader = PDFFormLoader(
    file_path='/Users/othmanezoheir/PycharmProjects/zoheir-consulting/lendmarq-ai/docs/Loan Application.pdf')
data = loader.load()
