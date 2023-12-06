from typing import Any, Type

from pydantic import BaseModel, ValidationError


class Validator(BaseModel):

    def __init__(self, **data: Any):
        if not data:
            raise ValidationError("At least one argument is required")
        super().__init__(**data)

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True
