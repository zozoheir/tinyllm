### API models

#### Function API model

Input:
```json
{
  "role": "user",
  "content": "Hi how are you"
}
```

Output:
```json
{
  "status": "success",
  "output": {
    // Processed output data
  }
}
```
In case of an error:
```json
{
  "status": "error",
  "message": "exception message (str)"
}
```

#### FunctionStream API model
Everythign is the same as Function, but the output format is different due to streaming. Each new chunk is validated with "output_validator"

```json
{
  "status": "success",
  "output": {
    "streaming_status": "streaming", # or completed
    "type": "assistant_response", # or tool
    "delta": {
      // Incremental data
    },
    "completion": {
      // Completion data 
    }
  }
}
```
Then, the last message yielded by the FunctionStream is the usual Function output:
{
  "status": "success",
  "output": {
    // Processed output data
  }
}
