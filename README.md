#### Architecture of the Library
The TinyLLM library consists of several key components that work together to facilitate the creation and management of Language Model Microservices (LLMs):
* **Function**: The base class for all LLM functions. It handles the execution of the LLM and the transition between different states in the LLM's lifecycle.
* **Validator**: A utility class used to validate input and output data for LLM functions.
* **Chain**: A function that allows the chaining of multiple LLM functions together to form a pipeline.
* **Decision**: A function that represents a decision point in the pipeline, allowing different paths to be taken based on the output of a previous LLM function.
* **Parallel**: A function that enables parallel execution of multiple LLM functions, useful for processing data concurrently.
* **OpenAIChat**: A function that interfaces with OpenAI's Chat API for text generation tasks.
* **OpenAIPromptTemplate**: A function that helps structure prompts for OpenAI's Chat API.
* **OpenAISystemMessage**: A function that represents a system message in a prompt.
* **OpenAIUserMessage**: A function that represents a user message in a prompt.
* **LLMCall**: A function for making API calls to external language model services.
