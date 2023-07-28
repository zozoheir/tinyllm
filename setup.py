from setuptools import setup, find_packages

setup(name='tiny-llm',
      version='0.1',
      description='Development and management infra for LLM applications',
      packages=find_packages(),
      entry_points={
            'console_scripts': [
                  'tinyllm = tinyllm.cli:main',
            ],
      },
      author='Othmane Zoheir',
      author_email='zozoheir@umich.edu',
      url='')
