"""
QuestionAnswerGenerator:
- input: documents, embedding function
- output: list of (context, question, truthful_answer) dicts

AnswerAccuracyEvaluator:
- inputs : question, truthful_answer, generated_answer
- outputs: accuracy, explanation

Context relevance:
- inputs: context, answer, generated_answer
- outputs: similarity

EvalPipeline:
- inputs: rag_lambda, evaluators, list of (context, question, truthful_answer)
- output: list of evaluator outputs

"""
from typing import List

from tinyllm.functions.function import Function

class EvaluationPipeline:

    def __init__(self,
                 rag_lambda,
                 qa_test_set,
                 evaluators: List[Function]):
        self.rag_lambda = rag_lambda
        self.qa_test_set = qa_test_set
        self.evaluators = evaluators

    async def run_evals(self):

        # Predict an answer for each question
        for data_point in self.qa_test_set:
            data_point["generated_answer"] = await self.rag_lambda(data_point["question"])

        # Run each evaluator
        for evaluator in self.evaluators:
            for data_point in self.qa_test_set:
                eval_result = await evaluator(context=data_point["context"],
                                              question=data_point["question"],
                                              truthful_answer=data_point["truthful_answer"],
                                              generated_answer=data_point["generated_answer"])
                data_point.update(eval_result)
        import pandas as pd
        df = pd.DataFrame(self.qa_test_set)
        return self.qa_test_set

