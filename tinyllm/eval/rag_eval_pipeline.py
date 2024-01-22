"""
QuestionAnswerGenerator:
- input: documents, embedding function
- output: list of (context, question, correct_answer) dicts

AnswerAccuracyEvaluator:
- inputs : question, correct_answer, generated_output
- outputs: accuracy, explanation

Context relevance:
- inputs: context, answer, generated_output
- outputs: similarity

EvalPipeline:
- inputs: rag_lambda, evaluators, list of (context, question, correct_answer)
- output: list of evaluator outputs

"""
from typing import List

from tinyllm.function import Function

class RagEvaluationPipeline:

    def __init__(self,
                 rag_lambda,
                 qa_test_set,
                 evaluators: List[Function] ):
        self.rag_lambda = rag_lambda
        self.qa_test_set = qa_test_set
        self.evaluators = evaluators

    async def run_evals(self):

        # Predict an answer for each question
        for data_point in self.qa_test_set:
            retrieved_chunks, generated_output, generation_id = await self.rag_lambda(data_point["question"])
            toinsert = {
                "retrieved_chunks": retrieved_chunks,
                "generated_answer": generated_output,
                "generation_id": generation_id
            }
            data_point.update(toinsert)

        # Run each evaluator
        for data_point in self.qa_test_set:
            data_point['scores'] = {}
            for evaluator in self.evaluators:
                eval_result = await evaluator(**data_point)
                if eval_result['status'] == 'success':
                    data_point['scores'].update(eval_result['output'])
                else:
                    data_point['scores'].update(eval_result)
        return self.qa_test_set