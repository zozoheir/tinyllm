import json
import traceback
from typing import Any

from tinyllm.function import Function
from tinyllm.llm_ops import langfuse_client
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

    # @fallback_decorator
    async def __call__(self, **kwargs):
        try:
            self.input = kwargs

            # Validate input
            self.transition(States.INPUT_VALIDATION)
            validated_input = self.validate_input(**kwargs)

            # Run
            self.transition(States.RUNNING)
            messages = []
            async for message in self.run(**validated_input):
                # we only validate the final output message
                msg = {
                    'streaming_status': 'streaming',
                    'type': message['type'],
                    'delta': message['delta'],
                    'completion': message['completion'],
                }

                # Output validation
                self.transition(States.OUTPUT_VALIDATION)
                self.validate_output(**msg)
                messages.append(msg)

                yield {"status": "success",
                       "output": msg}

            msg['streaming_status'] = 'completed'
            yield {"status": "success",
                   "output": msg}

            self.output = msg

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
            self.transition(States.FAILED, msg=traceback.format_exception(e))
            langfuse_client.flush()
            if type(e) in self.fallback_strategies:
                raise e
            else:
                yield {"status": "error",
                       "message": traceback.format_exception(e)}
