import os
import pprint
from typing import List, Union

import pandas as pd

from smartpy.cloud.do.storage import DigitalOcean
from smartpy.utility import os_util
from tinyllm import tinyllm_config
from tinyllm.llms.tiny_function import tiny_function
from tinyllm.rag.document.document import Document, ImageDocument
from tinyllm.util.message import UserMessage, Text, Image, Content


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


@tiny_function(model_params={'model': 'gpt-4-vision-preview'})
async def parse_image(content: Union[Content, List[Content]]):
    """
    <system>
    ROLE:
    You are an Image document parser. You will be provided an image and/or text content from a single document. Your goal is to extract
    structured data (sections, fields, descriptions) from this document using the provided img/text.

    OUTPUT FORMAT:
    Your output should be a JSON object that properly structures the extracted data. Make sure the section
    and field names are semantically meaningful.
    </system>
    """
    pass


class ImageStorageSources:
    DO = 'digital ocean'


class ImageLoader(Loader):

    def __init__(self,
                 file_path,
                 url: str = None,
                 content: str = None,
                 storage_source=None):
        super().__init__(file_path)
        self.url = url
        self.img_local_path = file_path
        self.storage_source = storage_source
        self.content = content

    def store_image(self):
        if self.url:
            return self.url

        if self.storage_source == ImageStorageSources.DO:
            do = DigitalOcean(
                region_name="nyc3",
                endpoint_url=tinyllm_config['CLOUD_PROVIDERS']['DO']['ENDPOINT'],
                key_id=tinyllm_config['CLOUD_PROVIDERS']['DO']['KEY'],
                secret_access_key=tinyllm_config['CLOUD_PROVIDERS']['DO']['SECRET']
            )
            self.url = do.upload_file(
                project_name=tinyllm_config['CLOUD_PROVIDERS']['DO']['PROJECT_NAME'],
                space_name="tinyllm",
                file_src=self.img_local_path,
                is_public=True
            )

        return self.url

    async def parse_with_ai(self):
        content = [
            Text("Use the provided img and its content to extract the relevant structured data and sections"),
            Image(self.url)
        ]
        parsing_output = await parse_image(content=content)
        return parsing_output['output'].dict()

    async def async_load(self, parse=False) -> Document:
        self.store_image()
        content = None
        if parse:
            content = str(self.parse_with_ai())
        image_doc = ImageDocument(
            content=content,
            url=self.url,
            metadata={})
        return image_doc


class PDFFormLoader(Loader):

    def __init__(self, file_path):
        super().__init__(file_path)
        from PyPDF2 import PdfReader
        self.pdf_reader = PdfReader(open(file_path, "rb"))

    async def async_load(self,
                         images=False) -> str:
        # Parse form dict
        form_content = self.pdf_reader.get_form_text_fields()
        form_content = {str(k): (float(v) if v.isdigit() else v) for k, v in form_content.items() if
                        v is not None and len(v) > 0}
        form_content = pprint.pformat(form_content)
        img_docs = []
        if images:
            screenshot_paths = self.get_screenshots(self.file_path)
            image_loaders = [ImageLoader(file_path=image_path,
                                         content=form_content,
                                         storage_source=ImageStorageSources.DO) for image_path in
                             screenshot_paths]
            img_docs = [await img_loader.async_load() for img_loader in image_loaders]

        pdf_form_doc = Document(content=form_content,
                                metadata={'urls': [img_doc.url for img_doc in img_docs]})

        return [pdf_form_doc]+img_docs

    def get_screenshots(self, pdf_path, dpi=500):
        from pdf2image import convert_from_path

        images_path = os_util.getTempDir('tinyllm/files/')
        base_name = '-'.join(os_util.getBaseName(pdf_path).split('.'))
        pages = convert_from_path(pdf_path, dpi)
        image_paths = []
        for count, page in enumerate(pages):
            file_name = base_name + f'_page_{count}.jpg'
            img_path = os.path.join(images_path, file_name)
            page.save(img_path, 'JPEG')
            image_paths.append(img_path)
        return image_paths


async def main():
    loader = PDFFormLoader(
        file_path='/Users/othmanezoheir/PycharmProjects/zoheir-consulting/lendmarq-ai/docs/Loan Application.pdf')
    doc = await loader.async_load(images=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
