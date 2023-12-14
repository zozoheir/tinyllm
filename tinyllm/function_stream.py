import traceback
from abc import abstractmethod
from typing import Any

from tinyllm.function import Function
from tinyllm import langfuse_client
from tinyllm.state import States
from tinyllm.validator import Validator


class DefaultFunctionStreamOutputValidator(Validator):
    streaming_status: str
    type: str  # assistant_response, tool
    delta: dict
    completion: Any


class FunctionStream(Function):

    def __init__(self,
                 **kwargs):
        super().__init__(output_validator=DefaultFunctionStreamOutputValidator,
                         **kwargs)

    @abstractmethod
    async def run(self,
                  **kwargs):
        yield None

    # @fallback_decorator
    async def __call__(self, **kwargs):
        try:
            self.input = kwargs

            # Validate input
            self.transition(States.INPUT_VALIDATION)
            validated_input = self.validate_input(**kwargs)

            # Run
            self.transition(States.RUNNING)
            async for message in self.run(**validated_input):

                # Output validation
                if 'status' in message.keys():
                    if message['status'] =='success':
                        message = message['output']
                    else:
                        raise Exception(message['message'])

                self.transition(States.OUTPUT_VALIDATION)
                self.validate_output(**message)

                yield {"status": "success",
                       "output": message}

            message['streaming_status'] = 'completed'
            yield {"status": "success",
                   "output": message}

            self.output = message

            # Process output
            self.transition(States.PROCESSING_OUTPUT)
            self.processed_output = await self.process_output(**self.output)

            # Validate processed output
            if self.processed_output_validator:
                self.processed_output = self.validate_processed_output(**self.processed_output)

            # Return final output
            final_output = {"status": "success",
                            "output": self.processed_output}

            yield final_output

            # Evaluate
            if self.evaluators:
                self.transition(States.EVALUATING)
                await self.evaluate(generation=self.generation,
                                    output=final_output,
                                    **kwargs)

            # Complete
            self.transition(States.COMPLETE)
            langfuse_client.flush()

        except Exception as e:
            self.error_message = str(e)
            self.transition(States.FAILED, msg='/n'.join(traceback.format_exception(e)))
            langfuse_client.flush()
            if type(e) in self.fallback_strategies:
                raise e
            else:
                yield {"status": "error",
                       "message": traceback.format_exception(e)}
