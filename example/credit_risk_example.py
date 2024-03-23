from pydantic import BaseModel
from tinyllm.llms.tiny_function import tiny_function
import random
import asyncio

# Define the output model for the Risk Score
class RiskScoreOutput(BaseModel):
    risk_score: float

@tiny_function(output_model=RiskScoreOutput)
async def calculate_risk_score(bank_account_history: str, employment_history: str):
    """
    <system>
    Extract a Risk Score between 0 and 1 for a Credit Card application based on bank account and employment history.
    </system>

    <prompt>
    Given the bank account history: {bank_account_history}
    And the employment history: {employment_history}
    Calculate the risk score for a credit card application.
    </prompt>
    """
    pass

# Generate random data for bank account and employment history
bank_account_history = 'Account balance over the last 12 months: ' + ', '.join([str(random.uniform(-2000, 10000)) for _ in range(12)])
employment_history = 'Employed for ' + str(random.randint(0, 30)) + ' years in ' + random.choice(['technology', 'education', 'healthcare', 'finance'])

# Run the async function
def main():
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(calculate_risk_score(bank_account_history=bank_account_history, employment_history=employment_history))
    print('Risk Score:', result['output'].risk_score)

if __name__ == '__main__':
    main()
