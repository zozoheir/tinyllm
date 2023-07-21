from pydantic import BaseModel
from typing import Any, List

class OperatorModel(BaseModel):
    name: str
    verbose: bool = False
    type: str = "operator"
    parent_id: str = None

class OperatorListModel(BaseModel):
    name: str
    children: List[OperatorModel]
    verbose: bool = False
    type: str = "operator_list"
    parent_id: str = None
