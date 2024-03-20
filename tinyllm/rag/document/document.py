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
                 content=None,
                 metadata: dict = {},
                 type=DocumentTypes.TEXT,
                 header='[doc]',
                 include_keys=['content']):
        self.content = content
        self.metadata = metadata
        self.type = type
        self.header = header
        self.include_keys = include_keys

    @property
    def size(self):
        content = self.to_string()
        return count_tokens(content)

    def to_dict(self):
        if self.type == DocumentTypes.TEXT:
            metacopy = self.metadata.copy()
            metacopy.update({'content': self.content})
            return metacopy
        elif self.type == DocumentTypes.DICTIONARY:
            return self.content

    def to_string(self):
        if self.type == DocumentTypes.TEXT:
            dic = self.to_dict()
            return stringify_dict(header=self.header,
                                  dict=dic,
                                  include_keys=self.include_keys)
        elif self.type == DocumentTypes.DICTIONARY:
            return stringify_dict(header=self.header,
                                  dict=self.content,
                                  include_keys=self.include_keys)
