# tinyllm
tinyllm is a lightweight framework for managing LLM powered applications at scale. It is designed based on a Finite State Machine and Compute graph model. 

#### Benefits
Many of the LLM libraries today (langchain, llama-index, deep pavlov...) have made serious software design commitments which I believe were too early to make given the infancy of the industry. The goal of tiny LLM is to 2 fold:
* Solve painpoints from current libraries: lack of composability (within + between libraries), complex software designs, code readability, debugging and logging.
* Stay as universal and general as possible, with the fewest lines of code and requirements as possible.

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


## Examples
*  [Generating jokes](https://github.com/zozoheir/tiny-llm/blob/main/tinyllm/examples/credit_analysis.py): a basic role/character
*  [Classifying a credit application](https://github.com/zozoheir/tiny-llm/blob/main/tinyllm/examples/credit_analysis.py): automating a business process with an LLM function call
*  **Graphing a chain in 1 line of code**:
```python
chain.graph()
```
![Figure_1](https://github.com/zozoheir/tiny-llm/assets/42655961/c49669dd-a1b1-4a9c-ab9c-2029628a6b3c)


