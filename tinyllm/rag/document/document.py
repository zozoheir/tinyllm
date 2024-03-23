from tinyllm.util.helpers import count_tokens
from tinyllm.util.prompt_util import stringify_dict


class Document:

    def __init__(self,
                 content=None,
                 metadata: dict = {}):
        self.content = content
        self.metadata = metadata
        self.type = type

    @property
    def size(self):
        content = self.to_string()
        return count_tokens(content)

    def to_string(self,
                  header,
                  include_keys):
        if type(self.content) == dict:
            content = stringify_dict(header=header,
                                     dict=self.content,
                                     include_keys=include_keys)
        else:
            content = self.content
        return content

class ImageDocument(Document):

    def __init__(self, url, **kwargs):
        super().__init__(**kwargs)
        self.url = url
