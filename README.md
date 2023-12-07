![Screenshot 2023-07-25 at 3 48 43 AM](https://github.com/zozoheir/tiny-llm/assets/42655961/f2db0c02-c18c-45a8-8054-6cd4da474e1e)

# 🕸️ tinyllm
tinyllm is a lightweight framework for developing, debugging and monitoring LLM powered applications at scale. It sits as a layer between your Web application and your LLM libraries.
`Function` and its streaming equivalent `FunctionStream` are designed to standardize LLM function calls for production use. 
It provides a structured approach to handle various aspects of function execution, including input/output validation, output processing, error handling, evaluation, all while maintaining high standards of code clarity and efficiency.

The goal of the library is to keep things simple and reduce the unnecessary complexity of libraries like Langchain, llama index etc...while allowing their integration easily if needed.

## Install
```
pip install git+https://github.com/zozoheir/tinyllm.git
```
## Features
- **litellm integration:** 20+ model providers available (OpenAI, Huggingface etc ...)
- **Prompt engineering:** utility modules for prompt engineering, optimization and string formatting
- **Layered validation:** 3 validations happen during the Function lifecycle: input, output and output processing.
- **IO Standardization:** Maintains consistent response patterns and failure handling across different function implementations.
- **Observability:** Integrates with Langfuse for monitoring and tracing of function execution at scale.
- **Memory:** conversations history 
- **Evaluation:** Evaluators can be defined to evaluate and log the quality of the function's output in real-time
- **PGVector store:** PostgreSQL DB with the pgvector extension for vector storage.
- **Logging:** Records detailed logs for debugging and auditing purposes.
- **Finite State Machine design:** Manages the function's lifecycle through defined states, ensuring controlled and predictable execution.

## API model
LLM Functions are designed to behave like a web API. All Functions take "role" and "content" as input arguments and will always, even if failed, return a dictionary response.

#### Validation
Validations are defined through a Pydantic model and are provided to the Function using input_validator, output_validator and output_processing_validator args to a Function
 
## How to and examples
* ####  [Check all examples here](https://github.com/zozoheir/tinyllm/blob/main/docs/examples.md)
## Function and FunctionStream API model
* ####  [Function and FunctionStream model](https://github.com/zozoheir/tinyllm/blob/main/docs/api_model.md)


## Background and goals
Many of the LLM libraries today (langchain, llama-index, deep pavlov...) have made serious software design commitments which I believe were too early to make given the infancy of the industry.
The goals of tinyllm are:
* **Solve painpoints from current libraries**: lack of composability (within + between libraries), complex software designs, code readability, debugging and logging.
* **High level, robust abstractions**: tinyllm is designed to be as simple as possible to use and integrate with existing and living codebases.
* **Human and machine readable code** to enable AI powered and autonomous chain development

## Tracing
tinyllm is integrated with Langfuse for tracing chains, functions and agents.
![Screenshot 2023-08-11 at 12 45 07 PM](https://github.com/zozoheir/tinyllm/assets/42655961/4d7c6ae9-e9a3-4795-9496-ad7905bc361e)


## ⚡ Classes
The TinyLLM library consists of several key components that work together to facilitate the creation and management of Language Model Microservices (LLMs):
* **Function**: The base class for all LLM functions. It handles the execution of the LLM and the transition between different states in the LLM's lifecycle.
* **FunctionStream**: The streaming equivalent of Function. 
* **Validator**: class to validate input and output data for LLM functions.
* **Evaluator**: class to evaluate the quality of the output of an LLM function.
* **VectorStore**: class to manage vector storage and search.

### Managing configs and credentials
Configs are managed through a tinyllm.yaml file. It gets picked up at runtime in tinyllm.__init__ and can be placed in any of /Documents, the root folder, or the current working directory. Here is a sample yaml config file:
```yaml
LLM_PROVIDERS:
  OPENAI_API_KEY: ""
LANGFUSE:
  LANGFUSE_PUBLIC_KEY: ""
  LANGFUSE_SECRET_KEY: ""
POSTGRES:
  TINYLLM_POSTGRES_USERNAME: ""
  TINYLLM_POSTGRES_PASSWORD: ""
  TINYLLM_POSTGRES_HOST: ""
  TINYLLM_POSTGRES_PORT: 
  TINYLLM_POSTGRES_NAME: ""
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

### Logging
The generation id is logged to quickly go from reading code to visualizating the chaining/conversation in Langfuse.

```
INFO | tinyllm.function | 2023-12-07 16:45:44,040 : [Test: LiteLLMChat_memory] transition to: States.PROCESSING_OUTPUT 
INFO | tinyllm.function | 2023-12-07 16:45:44,040 : [Test: LiteLLMChat_memory] transition to: States.PROCESSED_OUTPUT_VALIDATION 
INFO | tinyllm.function | 2023-12-07 16:45:44,040 : [Test: LiteLLMChat_memory] transition to: States.COMPLETE 
INFO | tinyllm.function | 2023-12-07 16:45:44,040 : [Test: LiteLLMChat|0a6c5186-8361-4245-b555-625a0595d744] transition to: States.OUTPUT_VALIDATION 
INFO | tinyllm.function | 2023-12-07 16:45:44,040 : [Test: LiteLLMChat|0a6c5186-8361-4245-b555-625a0595d744] transition to: States.PROCESSING_OUTPUT 
INFO | tinyllm.function | 2023-12-07 16:45:44,041 : [Test: LiteLLMChat|0a6c5186-8361-4245-b555-625a0595d744] transition to: States.PROCESSED_OUTPUT_VALIDATION 
INFO | tinyllm.function | 2023-12-07 16:45:44,041 : [Test: LiteLLMChat|0a6c5186-8361-4245-b555-625a0595d744] transition to: States.COMPLETE 
```
