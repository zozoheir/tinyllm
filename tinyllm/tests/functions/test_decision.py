import unittest

from tinyllm.tests.base import AsyncioTestCase
from tinyllm.functions.decision import Decision


class TestDecision(AsyncioTestCase):

    def test_decision_good_choice(self):
        choices = ['good', 'bad']
        async def decide(**kwargs):
            return {'decision': 'good'}

        decision = Decision(name="test-decision",
                            choices=choices,
                            run_function=decide)
        good_loan_application_example = "good file"
        result = self.loop.run_until_complete(decision(loan_application=good_loan_application_example))
        self.assertIsNotNone(result)
        self.assertTrue('decision' in result)
        self.assertIn(result['decision'], choices)
        self.assertEqual(result['decision'], 'good')


if __name__ == '__main__':
    unittest.main()
