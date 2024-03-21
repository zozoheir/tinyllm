from typing import List, Union, Dict


# Base class for content types
class Content:
    def dict(self) -> Dict:
        return self.__dict__


class Text(Content):
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class ImageUrl(Content):
    def __init__(self, url: str):
        self.type = "image_url"
        self.image_url = {"url": url}


# Base class for message roles
class Message:
    def __init__(self, role: str, content: List[Content]):
        self.role = role
        self.content = content

    def dict(self) -> Dict:
        return {"role": self.role, "content": [c.dict() for c in self.content]}


class UserMessage(Message):
    def __init__(self, content: List[Content], **kwargs):
        super().__init__("user", content, **kwargs)


class SystemMessage(Message):
    def __init__(self, content: List[Content], **kwargs):
        super().__init__("system", content, **kwargs)


# Additional roles as needed
class FunctionMessage(Message):
    def __init__(self, content: List[Content], **kwargs):
        super().__init__("function", content, **kwargs)


class ToolMessage(Message):
    def __init__(self, content: List[Content], **kwargs):
        super().__init__("tool", content, **kwargs)


class AssistantMessage(Message):
    def __init__(self, content: List[Content], **kwargs):
        super().__init__("assistant", content, **kwargs)

