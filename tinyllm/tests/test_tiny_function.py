from pydantic import BaseModel, Field

from tinyllm.llms.tiny_function import tiny_function
from tinyllm.tests.base import AsyncioTestCase
from tinyllm.util.message import Text

# Mock response to simulate Agent's response
mock_success_response = {
    "status": "success",
    "output": {
        "response": {
            "choices": [{
                "message": {
                    "content": "{\"name\": \"Elon\", \"age\": 50, \"occupation\": \"CEO\"}"
                }
            }]
        }
    }
}

mock_fail_response = {"status": "error", "response": {"message": "Failed to process request"}}


class TestTinyFunctionDecorator(AsyncioTestCase):

    def test_tiny_function_success(self):
        class CharacterInfo(BaseModel):
            name: str = Field(..., description="Name")
            age: int = Field(..., description="Age")
            occupation: str = Field(..., description="occupation")

        @tiny_function(output_model=CharacterInfo)
        async def get_character_info(doc1, doc2):
            """
            <system>
            Extract character information from the provided documents
            </system>

            <prompt>
            {doc1}
            {doc2}
            </prompt>
            """
            pass

        # Test the decorated function
        content = "Elon Musk is a 50 years old CEO"
        result = self.loop.run_until_complete(get_character_info(doc1=content, doc2=content))

        # Assertions
        self.assertIsInstance(result['output'], dict)
        self.assertTrue("Elon" in result['output']['name'])
        self.assertTrue(result['output']['age'], 50)
        self.assertTrue("CEO" in result['output']['occupation'])

    def test_no_model(self):
        @tiny_function()
        async def get_character_info(content: str):
            """
            <system>
            Extract character information from the content
            </system>
            """
            pass

        # Test the decorated function
        content = "Elon Musk is a 50 years old CEO"
        result = self.loop.run_until_complete(get_character_info(content=content))
        self.assertEqual(result['status'], 'success')

    def test_not_enough_tokens(self):
        class CharacterInfo(BaseModel):
            name: str = Field(..., description="Name")
            age: int = Field(..., description="Age")
            occupation: str = Field(..., description="occupation")

        @tiny_function(model_kwargs={'max_tokens': 80}, output_model=CharacterInfo)
        async def get_character_info(content: str):
            """
            <system>
            Extract character information from the content
            </system>
            """
            pass

        # Test the decorated function
        content = "Elon Musk is a 50 years old CEO"
        result = self.loop.run_until_complete(get_character_info(content=content))
        print(result)
        self.assertEqual(result['status'], 'success')
