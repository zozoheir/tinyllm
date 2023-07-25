![Screenshot 2023-07-25 at 2 41 40 AM](https://github.com/zozoheir/tiny-llm/assets/42655961/73ab8f68-faaf-4bda-96cb-0703bc8a911a)

# 🕸️ tinyllm
tinyllm is a lightweight framework for developing, debugging and monitoring LLM powered applications at scale. It is designed based on a Finite State Machine and Compute graph model. 

## Goal of the library
Many of the LLM libraries today (langchain, llama-index, deep pavlov...) have made serious software design commitments which I believe were too early to make given the infancy of the industry. The goal of tiny LLM is to 2 fold:
* Solve painpoints from current libraries: lack of composability (within + between libraries), complex software designs, code readability, debugging and logging.
* Stay as universal and general as possible, with the fewest lines of code and requirements as possible.
* Make code as readable and writable as possible for LLMs to enable AI powered and autonomous chain development

## ⚡ Features
* Integrate tiny-llm with any LLM library or existing python code or pipelines
* Compose, debug and track LLM calls and chains at scale
* Visualize chains in 1 line of code
* High level abstraction of LLM/API chaining/interactions through a standardized I/O interface

## Architecture
The TinyLLM library consists of several key components that work together to facilitate the creation and management of Language Model Microservices (LLMs):
* **Function**: The base class for all LLM functions. It handles the execution of the LLM and the transition between different states in the LLM's lifecycle.
* **Validator**: A utility class used to validate input and output data for LLM functions.
* **Chain**: A function that allows the chaining of multiple LLM functions together to form a pipeline of calls.
* **Concurrent**: A function that enables concurrent execution of multiple LLM functions, useful for ensembling/comparing from different LLMs or speeding up IO bound task execution.
* **Decision**: A function that represents a decision point in the chain, allowing different paths to be taken based on the output of a previous LLM function.
* **LLMCall**: A function for making API calls to external language model services.
* **Prompt**: A function for generating prompts from templates and user inputs.


## Concurrent vs Parallel execution
This tends to be confusing vocabulary but let's clarify it for this context:
- Parallel execution: This means more than 1 Python process running at a time. This means compute/calculations are being performed on more than 1 process on the same machine. This is what model providers do using large GPU clusters (Nvidia, AMD...). This is used for "CPU Bound" tasks.
- Concurrent execution: This means more than 1 IO request (to the web) at a time. Just like you can download 10 files on your web browser, you can call 10 APIs concurrently.

A good design for the LLM layer of an application would be an LLM models microservice. This allows abstracting away the design choices of LLM providers (OpenAI, hugging face...) and LLM libraries and have a single IO interface to call any tinyllm LLMCall. This interface will be used for logging, caching, data persistence and monitoring.

## Examples
*  [Generating jokes](https://github.com/zozoheir/tiny-llm/blob/main/tinyllm/examples/credit_analysis.py): a basic role/character
*  [Classifying a credit application](https://github.com/zozoheir/tiny-llm/blob/main/tinyllm/examples/credit_analysis.py): automating a business process with an LLM function call
*  **Graphing a chain in 1 line of code**:
```python
chain.graph()
```
![Figure_1](https://github.com/zozoheir/tiny-llm/assets/42655961/c49669dd-a1b1-4a9c-ab9c-2029628a6b3c)


## Todos:
* [ ] More tests
* [ ] Add .from_params() method to Functions for easy initialization
* [ ] Prettify graph visualization (concurrent vs parallel chaining + styling)
* [ ] Implement backend database 
* [ ] Implement redis caching
* [ ] Dockerize backend db + cache
* [ ] Dockerize tinyllm microservice
* [ ] Write docker compose
* [ ] Implement visualization/monitoring layer
* [ ] Create tinyllm trained AI powered helpers
