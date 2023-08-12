import unittest

from tinyllm.tests.base import AsyncioTestCase
from tinyllm.functions.chain import Chain
from tinyllm.functions.decision import Decision
from tinyllm.functions.function import Function
from tinyllm.functions.concurrent import Concurrent
from tinyllm.state import States


class TestCreditAnalysis(AsyncioTestCase):

    def test_credit_chain(self):
        # Initialize a Loan Classifier OpenAI Prompt template
        async def classify_loan_application(**kwargs):
            loan_application = kwargs.get('loan_application')
            print(f"Credit classification:good")
            return {'decision': 'good'}

        loan_classifier = Decision(
            name="Decision: Loan classifier",
            choices=['good', 'bad'],
            run_function=classify_loan_application,
            is_traced=True
        )

        async def send_email(**kwargs):
            print("Sending email notification...")
            return {'success': True}

        email_notification = Function(
            name="Email notification",
            run_function=send_email
        )

        async def background_check(**kwargs):
            print("Performing background get_check_results")
            return {'background_check': 'Completed'}

        bg_check = Function(
            name="Background get_check_results",
            run_function=background_check,
        )


        credit_analysis_chain = Chain(name="Chain: Loan application",
                                      children=[
                                          loan_classifier,
                                          Concurrent(name="Concurrent: On good credit",
                                                     children=[email_notification,
                                                               bg_check])],
                                      is_traced=True)
        result = self.loop.run_until_complete(credit_analysis_chain(loan_application="example"))

        for node, state in credit_analysis_chain.graph_state.items():
            self.assertEqual(state, States.COMPLETE)


if __name__ == '__main__':
    unittest.main()

