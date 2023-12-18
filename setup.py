from setuptools import setup, find_packages

setup(name='tinyllm',
      version='1.0.10',
      description='Development and management infrastructure for LLM applications',
      packages=find_packages(),
      install_requires=[
          'click',
          'tenacity',
          'langfuse==1.14.0',
          'langchain',
          'litellm',
          'openai',
          'pathspec',
          'pgvector',
          'psutil',
          'psycopg2-binary',
          'pydantic==1.10.7',
          'pytest',
          'pyyaml',
          'sqlalchemy',
          'tiktoken',
          'typing-extensions==4.5.0',
          'uuid',
          'fuzzywuzzy'
      ],
      author='Othmane Zoheir',
      author_email='zozoheir@umich.edu',
      url='')
