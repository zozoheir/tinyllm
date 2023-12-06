import traceback

from tinyllm.function import Function
from tinyllm.state import States


class FunctionStream(Function):

    # @fallback_decorator
    async def __call__(self, **kwargs):
        try:
            self.input = kwargs
            self.transition(States.INPUT_VALIDATION)
            validated_input = await self.validate_input(**kwargs)
            self.transition(States.RUNNING)
            async for message in self.run(**validated_input):
                yield message
            self.output = {'response': message}
            self.transition(States.OUTPUT_VALIDATION)
            self.output = await self.validate_output(**self.output)
            self.transition(States.PROCESSING_OUTPUT)
            self.output = await self.process_output(**self.output)
            self.processed_output = self.output
            if self.evaluators:
                self.transition(States.EVALUATING)
                await self.evaluate(**kwargs)
            self.transition(States.COMPLETE)
        except Exception as e:
            self.error_message = str(e)
            self.transition(States.FAILED, msg=traceback.format_exception(e))
            if type(e) in self.fallback_strategies:
                raise e
