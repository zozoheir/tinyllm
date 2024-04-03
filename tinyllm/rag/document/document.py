from enum import Enum

from tinyllm.util.helpers import count_tokens
from tinyllm.util.prompt_util import stringify_dict


class DocumentTypes(Enum):
    TEXT = 'text'
    DICTIONARY = 'dictionary'
    TABLE = 'table'
    IMAGE = 'image'


class Document:

    def __init__(self,
                 content,
                 metadata: dict={},
                 embeddings = None,
                 type=DocumentTypes.TEXT,
                 header='[doc]',
                 include_keys=['content']):
        self.content = content
        self.metadata = metadata
        self.type = type
        self.header = header
        self.include_keys = include_keys
        self.embeddings = embeddings

    @property
    def size(self):
        content = self.to_string()
        return count_tokens(content)

    def to_string(self,
                  **kwargs):
        full_dict = self.metadata.copy()
        full_dict.update({'content': self.content})
        return stringify_dict(header=kwargs.get('header', self.header),
                              dict=full_dict,
                              include_keys=kwargs.get('include_keys', self.include_keys))


class ImageDocument(Document):

    def __init__(self, url, **kwargs):
        super().__init__(**kwargs)
        self.url = url
