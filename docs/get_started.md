## Getting started with tinyllm

## Configs


A yaml config file needs to be provided as an environment variable. [**The config file template is is available here**](https://github.com/zozoheir/tinyllm/blob/main/docs/get_started.md)

```python
import os
os.environ['TINYLLM_CONFIG_PATH'] = '/path/to/tinyllm.yaml'
```


If the config file is not provided, the library will look for a tinyllm.yaml config file in the following directories:
- The current working directory
- The user's home directory
- The user's Document's directory


## Example agent


```python
import asyncio 
from typing import Any, Optional
from pydantic import BaseModel, Field
from tinyllm.agent.agent import Agent

class Person(BaseModel):
    name: str = Field(..., description='Name of the person')
    age: int = Field(..., description='Age of the person')
    note: Optional[Any]

class RiskScoreOutput(BaseModel):
    risk_score: float = Field(..., description='A risk score between 0 and 1')
    person: Person

tiny_agent = Agent(
    name='Test: Agent JSON output',
    system_role="You are a Credit Risk Analyst. Respond with a risk score based on the provided customer data",
    json_pydantic_model=RiskScoreOutput
)

result = asyncio.run(tiny_agent(content="Johny Vargas, 29yo, the customer has missed 99% of his bill payments in the last year"))


```
