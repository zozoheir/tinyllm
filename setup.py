from setuptools import setup, find_packages

setup(name='tinyllm',
      version='0.1',
      description='Development and management infrastructure for LLM applications',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'tinyllm = tinyllm.cli:main',
          ],
      },
      install_requires=[
          'click',
          'tenacity',
          'gradio',
          'langchain==0.0.237',
          'langfuse==0.0.66',
          'openai==0.27.8',
          'pathspec',
          'pgvector',
          'psutil',
          'psycopg2-binary',
          'py2neo==2021.2.3',
          'pydantic==1.10.7',  # Compatible with langchain + langfuse
          'pytest==7.4.0',
          'pyyaml',
          'sqlalchemy',
          'tiktoken',
          'typing-extensions==4.5.0',  # This fixes the pydantic/subclass typing bug
          'uuid'
      ],
      author='Othmane Zoheir',
      author_email='zozoheir@umich.edu',
      url='')
