"""
QuestionAnswerGenerator:
- input: documents, embedding function
- output: list of (context, question, correct_answer) dicts

AnswerAccuracyEvaluator:
- inputs : question, correct_answer, generated_answer
- outputs: accuracy, explanation

Context relevance:
- inputs: context, answer, generated_answer
- outputs: similarity

EvalPipeline:
- inputs: rag_lambda, evaluators, list of (context, question, correct_answer)
- output: list of evaluator outputs

"""
from typing import List

from tinyllm.functions.function import Function

class RagEvaluationPipeline:

    def __init__(self,
                 rag_lambda,
                 qa_test_set,
                 evaluators: List[Function],
                 metadata={}):
        self.rag_lambda = rag_lambda
        self.qa_test_set = qa_test_set
        self.evaluators = evaluators
        self.metadata = metadata

    async def run_evals(self):

        # Predict an answer for each question
        for data_point in self.qa_test_set:
            generated_answer, generation_id = await self.rag_lambda(data_point["question"])
            data_point.update({
                "generated_answer": generated_answer,
                "generation_id": generation_id
            })

        # Run each evaluator
        for data_point in self.qa_test_set:
            data_point['scores'] = {}
            for evaluator in self.evaluators:
                eval_result = await evaluator(context=data_point["context"],
                                              question=data_point["question"],
                                              correct_answer=data_point["correct_answer"],
                                              generated_answer=data_point["generated_answer"])
                data_point['scores'].update(eval_result)
        return self.qa_test_set

