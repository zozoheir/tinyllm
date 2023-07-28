# 🕸️ tinyllm (beta)
tinyllm is a lightweight framework for developing, debugging and monitoring LLM powered applications at scale. It is designed based on a Finite State Machine and Compute Graph model and sits as a layer between your Web application and your LLM libraries. tinyllm has main components:
- the tinyllm library + CLI commands
- a neo4j backend database
- a tinyllm agent UI

![Screenshot 2023-07-25 at 3 48 43 AM](https://github.com/zozoheir/tiny-llm/assets/42655961/f2db0c02-c18c-45a8-8054-6cd4da474e1e)
![Screenshot 2023-07-28 at 2 19 48 AM](https://github.com/zozoheir/tinyllm/assets/42655961/61c8121e-0909-473e-a475-20626cf6452f)

## Install
```
pip install git+https://github.com/zozoheir/tinyllm.git
```

## Tinyllm agent v0
The start command will open a UI to the library's agent. 
```
tinyllm agent
```
![Screenshot 2023-07-28 at 2 06 05 AM](https://github.com/zozoheir/tinyllm/assets/42655961/7c5a9d62-4c79-499c-9d85-8a9a4a285190)

## ⚡ Features
* Integrate tiny-llm with any LLM library or existing python code or pipelines
* Compose, debug and track LLM calls and chains at scale
* Visualize chains in 1 line of code
* High level abstraction of LLM/API chaining/interactions through a standardized I/O interface

## ⚡ Background and goals
Many of the LLM libraries today (langchain, llama-index, deep pavlov...) have made serious software design commitments which I believe were too early to make given the infancy of the industry.
The goals of tinyllm are:
* **Solve painpoints from current libraries**: lack of composability (within + between libraries), complex software designs, code readability, debugging and logging.
* **High level, robust abstractions**: tinyllm is designed to be as simple as possible to use and integrate with existing and living codebases.
* **Human and machine readable code** to enable AI powered and autonomous chain development

## ⚡ LLMs as Finite State Compute graphs
tinyllm is based on the following principles:
* A Function has Finite States
* LLM calls are Compute graphs

Since an LLM Call is itself a Function, LLM chains are thus Finite State Compute graphs that can be tracked, logged and debugged as such. While Graphs have nice properties, they need have a specific way to be implemented from a software perspective (data storage, visualization, logging etc...).
On top of being Compute graphs, LLM Chains calls can be non deterministic (1 input can have more than 1 output) and interact through natural language. This is a new and unique paradigm for production software that hasn't been fully figured out yet. This is where tinyllm comes in.

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

## ⚡ Codebase
The TinyLLM library consists of several key components that work together to facilitate the creation and management of Language Model Microservices (LLMs):
* **Function**: The base class for all LLM functions. It handles the execution of the LLM and the transition between different states in the LLM's lifecycle.
* **Validator**: A utility class used to validate input and output data for LLM functions.
* **Chain**: A function that allows the chaining of multiple LLM functions together to form a pipeline of calls.
* **Concurrent**: A function that enables concurrent execution of multiple LLM functions, useful for ensembling/comparing from different LLMs or speeding up IO bound task execution.
* **Decision**: A function that represents a decision point in the chain, allowing different paths to be taken based on the output of a previous LLM function.
* **LLMCall**: A function for making API calls to external language model services.
* **Prompt**: A function for generating prompts from templates and user inputs.


### Tinyllm configs
Configs are managed through a tinyllm.yaml file. It gets picked up at runtime and can be placed in any of /Documents, the root folder, or the current working directory. Here is a sample yaml config file:
```yaml
LLM_PROVIDERS:
  OPENAI_API_KEY: 
DB:
  TINYLLM_DB_HOST: 
  TINYLLM_DB_PORT: 
  TINYLLM_DB_USER: 
  TINYLLM_DB_PASSWORD: 

DB_FUNCTIONS_LOGGING:
  DEFAULT: true
  INCLUDE:
    - 'CustomFunctionClass'
  EXCLUDE:
    - 'Function'
```
* DB_FUNCTIONS_LOGGING is used for including/excluding logging to the backend DB for certain classes like base classes (Function).
* DB is used for your neo4j backend
* tinyllm.__init__ has an App class that is used for loading the config file, connecting to the database etc. 



### Logging
The app has a single logger, single format, that can push logs to many different streams (console, file, custom...)
The default logger is managed through the App() object. 
Additionally, you can enable or disable logging of States on a Function level using 
```
    openai_chat = OpenAIChat(name='OpenAI-GPT model',
                             llm_name='gpt-3.5-turbo',
                             temperature=0,
                             n=1,
                             verbose=True)
## ⚡ Examples

### Chaining
* ####  [Classifying a credit application](https://github.com/zozoheir/tiny-llm/blob/main/tinyllm/examples/credit_analysis.py)
### Chat
* ####  [A basic openaichat](https://github.com/zozoheir/tinyllm/blob/main/tinyllm/examples/openai_chat.py)
### Retrieval
* ####  [The tinyllm agent](https://github.com/zozoheir/tinyllm/blob/main/tinyllm/agent.py)

## Todos:
* [ ] More tests
* [ ] Prettify graph_chain visualization (concurrent vs parallel chaining + styling)
* [ ] Implement backend database 
* [ ] Implement redis caching
* [ ] Dockerize backend db + cache
* [ ] Dockerize tinyllm microservice
* [ ] Write docker compose
* [ ] Implement visualization/monitoring layer
* [ ] Create tinyllm trained AI powered helpers
