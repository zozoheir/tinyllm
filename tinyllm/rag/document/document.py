from enum import Enum

from tinyllm.util.helpers import count_tokens
from tinyllm.util.prompt_util import stringify_dict


class DocumentTypes(Enum):
    TEXT = 'text'
    DICTIONARY = 'dictionary'
    TABLE = 'table'


class Document:

    def __init__(self,
                 content,
                 metadata,
                 type=DocumentTypes.TEXT,
                 header='[doc]',
                 ignore_keys=[]):
        self.content = content
        self.metadata = metadata
        self.type = type
        self.header = header
        self.ignore_keys = ignore_keys

    @property
    def size(self):
        content = self.format()
        return count_tokens(content)

    def format(self):
        if self.type == DocumentTypes.TEXT:
            return self.content
        elif self.type == DocumentTypes.DICTIONARY:
            return stringify_dict(header=self.header,
                                  dict=self.content,
                                  ignore_keys=self.ignore_keys)
