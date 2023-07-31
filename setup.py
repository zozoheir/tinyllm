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
          'pytest==7.4.0',
          'pydantic==1.10.12',
          'pgvector'
          'openai==0.27.8',
          'tiktoken',
          'tiktoken',
          'py2neo==2021.2.3',
          'pyyaml',
          'gradio',
          'psutil',
          'click',
          'psycopg2-binary',
          'sqlalchemy',
          'uuid',
          'pathspec',
          'cohere'
      ],
      author='Othmane Zoheir',
      author_email='zozoheir@umich.edu',
      url='')
