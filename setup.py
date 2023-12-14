from setuptools import setup, find_packages

setup(name='tinyllm',
      version='1.0.2',
      description='Development and management infrastructure for LLM applications',
      packages=find_packages(),
      install_requires=[
          'click',
          'tenacity',
          'langchain',
          'langfuse',
          'litellm',
          'openai',
          'pathspec',
          'pgvector',
          'psutil',
          'psycopg2-binary',
          'pydantic==1.10.7',  # Compatible with langchain + langfuse
          'pytest',
          'pyyaml',
          'sentence_transformers',
          'sqlalchemy',
          'tiktoken',
          'typing-extensions==4.5.0',  # This fixes the pydantic/subclass typing bug
          'uuid',
          'fuzzywuzzy'
      ],
      author='Othmane Zoheir',
      author_email='zozoheir@umich.edu',
      url='')
