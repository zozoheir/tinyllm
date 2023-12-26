## Getting started with tinyllm

## Configs


A yaml config file needs to be provided as an environment variable.
```python
import os
os.environ['TINYLLM_CONFIG_PATH'] = '/path/to/tinyllm.yaml'
```

**An example config file is available in the repo source as tinyllm.yaml**

If the config file is not provided, the library will look for a tinyllm.yaml config file in the following directories:
- The current working directory
- The user's home directory
- The user's Document's directory


