<p align="center">
    <img src="https://github.com/zozoheir/tinyllm/assets/42655961/28d13a6a-8366-497f-b3e8-8be262f5b9fd" alt="tinyllm arc">
</p>



# 🚀 What is tinyllm?
tinyllm is a lightweight framework for developing, debugging and monitoring LLM and Agent powered applications at scale. The main goal of the library is to keep code as simple and readable as possible while allowing user to create complex agents or LLM workflows in production.

`Function` and its streaming equivalent `FunctionStream` are the core classes in tinyllm. They are designed to standardize and control LLM, ToolStore and any relevant calls for scalable production use in stream mode and otherwise.

It provides a structured approach to handle various aspects of function execution, including input/output validation, output processing, error handling, evaluation, all while keeping code readable. You can create a chain with its own prompt, LLM model and evaluators all in a single file. No need to jump through many class definitions, no spaghetti code. Any other library agent/chain (langchain/llama-index...) can also seamlessly be imported as a tinyllm Function.


## 🚀 Install
```
pip install tinyllm
```

## 🚀 Getting started
* ####  [Setup](https://github.com/zozoheir/tinyllm/blob/main/docs/setup.md)
* ####  [Examples](https://github.com/zozoheir/tinyllm/blob/main/docs/examples/)


## 🚀 Features
#### Build LLM apps with:
- **LiteLLM integration**: 20+ model providers available (OpenAI, Huggingface etc ...)
- **Langfuse integration**: Monitor trace and debug LLMs, Agents, Tools, RAG pipelines etc in structured run trees
- **Agents:** An agent is an LLM with Memory, a Toolkit and an ExampleManager
- **ToolStore and Toolkits**: let your Agent run python functions using ToolStore
- **Example manager**: constant examples + variable examples using and example selector with similarity search
- **Memory:** conversations history
- **Retrieval Augmented Generation**: RAG tools to search and generate answers
- **Evaluation:** Evaluators can be defined to evaluate and log the quality of the function's output in real-time
- **PGVector store:** PostgreSQL DB with the pgvector extension for vector storage.
- **Prompt engineering tools:** utility modules for prompt engineering, optimization and string formatting

#### 🚀 Deploy to production with:
- **Layered validation:** 3 validations happen during the Function lifecycle: input, output and output processing.
- **IO Standardization:** Maintains consistent response patterns and failure handling across different function implementations.
- **Observability:** Integrates with Langfuse for
- **Logging:** Records detailed logs for debugging and auditing purposes.
- **Finite State Machine design:** Manages the function's lifecycle through defined states, ensuring controlled and predictable execution.

#### Tiny function wrapper

```python
class RiskScoreOutput(BaseModel):
    risk_score: float

@tiny_function(output_model=RiskScoreOutput)
async def calculate_risk_score(bank_account_history: str, employment_history: str):
    """
    <system>
    Extract a Risk Score between 0 and 1 for a Credit Card application based on bank account and employment history.
    </system>

    <prompt>
    Given the bank account history: {bank_account_history}
    And the employment history: {employment_history}
    Calculate the risk score for a credit card application.
    </prompt>
    """
    pass
```

#### Tracing with Langfuse

<p align="center">
    <img src="https://github.com/zozoheir/tinyllm/assets/42655961/0284f94d-7c5d-4abb-900d-3fc3381d61dc" width="70%" height="70%" alt="initialize">
</p>

## Background and goals
Many of the LLM libraries today (langchain, llama-index, deep pavlov...) have made serious software design commitments which I believe were too early to make given the infancy of the industry.
The goals of tinyllm are:
* **Solve painpoints from current libraries**: lack of composability (within + between libraries), complex software designs, code readability, debugging and logging.
* **High level, robust abstractions**: tinyllm is designed to be as simple as possible to use and integrate with existing and living codebases.
* **Human and machine readable code** to enable AI powered and autonomous chain development

## API model
LLM Functions are designed to behave like a web API. All Functions will always, even if failed, return a dictionary response.

#### Validation
Validations are defined through a Pydantic model and are provided to the Function using input_validator, output_validator and output_processing_validator args to a Function

## Tracing
tinyllm is integrated with Langfuse for tracing chains, functions and agents.
![Screenshot 2023-08-11 at 12 45 07 PM](https://github.com/zozoheir/tinyllm/assets/42655961/4d7c6ae9-e9a3-4795-9496-ad7905bc361e)

### Managing configs and credentials
Configs are managed through a tinyllm.yaml file. It gets picked up at runtime in tinyllm.__init__ and can be placed in any of /Documents, your root folder, or the current working directory. 
An empty tinyllm.yaml file is at the source of the repo to get you setup.


## ⚡ Concurrency vs Parallelism vs Chaining
These tend to be confusing across the board. Here's a quick explanation:
- **Concurrency** : This means more than 1 Input/Ouput request at a time. Just like you can download 10 files 
concurrently on your web browser, you can call 10 APIs concurrently.
- **Chaining** : An ordered list of Functions where a Function's output is the input of the next Function in the chain.
- **Parallelism** : compute/calculations being performed on more than 1 process/CPU Core on the same machine. This is what 
model providers like OpenAI do using large GPU clusters (Nvidia, AMD...). This is used for "CPU Bound" tasks.

Tinyllm does not care about Parallelism. Parallelism is implemented by LLM providers
on a GPU/CPU level and should be abstracted away using an LLM microservice.
Tinyllm only cares about Concurrency, Chaining and organizing IO Bound tasks.


 
### Logging

Finite state machine with predictable and controlled state transitions for easy debugging of your chains/compute graphs.

Below is the start and end of a trace for asking "What is the weather in Puerto Rico?" to an Agent with a get_weather Tool. 

```
INFO | tinyllm.function | 2023-12-25 19:37:10,617 : [Standard example selector] transition to: States.INIT 
INFO | tinyllm.function | 2023-12-25 19:37:12,720 : [BufferMemory] transition to: States.INIT 
INFO | tinyllm.function | 2023-12-25 19:37:12,729 : [get_weather] transition to: States.INIT 
INFO | tinyllm.function | 2023-12-25 19:37:12,729 : [Toolkit] transition to: States.INIT 
INFO | tinyllm.function | 2023-12-25 19:37:12,731 : [LiteLLM] transition to: States.INIT 
...
...
INFO | tinyllm.function | 2023-12-25 19:37:17,150 : [AnswerCorrectnessEvaluator] transition to: States.PROCESSING_OUTPUT 
INFO | tinyllm.function | 2023-12-25 19:37:17,151 : [AnswerCorrectnessEvaluator] transition to: States.PROCESSED_OUTPUT_VALIDATION 
INFO | tinyllm.function | 2023-12-25 19:37:17,151 : [AnswerCorrectnessEvaluator] transition to: States.COMPLETE 
INFO | tinyllm.function | 2023-12-25 19:37:17,846 : [Agent] transition to: States.PROCESSING_OUTPUT 
INFO | tinyllm.function | 2023-12-25 19:37:17,847 : [Agent] transition to: States.PROCESSED_OUTPUT_VALIDATION 
INFO | tinyllm.function | 2023-12-25 19:37:17,847 : [Agent] transition to: States.COMPLETE 
{'status': 'success', 'output': {'response': {'id': 'chatcmpl-8ZpjY0QmXbDiMIcSRwKuCUny4sxul', 'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': "It is 25 degrees celsius in Puerto Rico", 'role': 'assistant'}}], 'created': 1703551035, 'model': 'gpt-3.5-turbo-0613', 'object': 'chat.completion', 'system_fingerprint': None, 'usage': {'completion_tokens': 12, 'prompt_tokens': 138, 'total_tokens': 150}, '_response_ms': 785.606}}}
```


## ⚡ Concurrency vs Parallelism vs Chaining
These tend to be confusing across the board. Here's a quick explanation:
- **Concurrency** : This means more than 1 Input/Ouput request at a time. Just like you can download 10 files 
concurrently on your web browser, you can call 10 APIs concurrently.
- **Chaining** : An ordered list of Functions where a Function's output is the input of the next Function in the chain.
- **Parallelism** : compute/calculations being performed on more than 1 process/CPU Core on the same machine. This is what 
model providers like OpenAI do using large GPU clusters (Nvidia, AMD...). This is used for "CPU Bound" tasks.

Tinyllm does not care about Parallelism. Parallelism is implemented by LLM providers
on a GPU/CPU level and should be abstracted away using an LLM microservice.
Tinyllm only cares about Concurrency, Chaining and organizing IO Bound tasks.


