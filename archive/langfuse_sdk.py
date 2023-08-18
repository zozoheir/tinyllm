import aiohttp
import json


class LangfuseClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.auth = aiohttp.BasicAuth(login=username, password=password)

    async def create_event(self, id, traceId, traceIdType, name, startTime, metadata, input, output, level,
                           statusMessage, parentObservationId, version):
        url = f"{self.base_url}/api/public/events"
        data = {
            "id": id,
            "traceId": traceId,
            "traceIdType": traceIdType,
            "name": name,
            "startTime": startTime,
            "metadata": metadata,
            "input": input,
            "output": output,
            "level": level,
            "statusMessage": statusMessage,
            "parentObservationId": parentObservationId,
            "version": version
        }
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    # Implement other methods in a similar way...

    async def create_trace(self, name, userId, externalId, release, version, metadata):
        url = f"{self.base_url}/api/public/traces"
        data = {
            "name": name,
            "userId": userId,
            "externalId": externalId,
            "release": release,
            "version": version,
            "metadata": metadata
        }
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    async def create_generation(self, id, traceId, traceIdType, name, startTime, endTime, completionStartTime, model, modelParameters, prompt, completion, usage, level, statusMessage, parentObservationId, version):
        url = f"{self.base_url}/api/public/generations"
        data = {
            "id": id,
            "traceId": traceId,
            "traceIdType": traceIdType,
            "name": name,
            "startTime": startTime,
            "endTime": endTime,
            "completionStartTime": completionStartTime,
            "model": model,
            "modelParameters": modelParameters,
            "prompt": prompt,
            "completion": completion,
            "usage": usage,
            "level": level,
            "statusMessage": statusMessage,
            "parentObservationId": parentObservationId,
            "version": version
        }
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    async def update_generation(self, generationId, traceId, name, endTime, completionStartTime, model, modelParameters, prompt, version, metadata, completion, usage, level, statusMessage):
        url = f"{self.base_url}/api/public/generations"
        data = {
            "generationId": generationId,
            "traceId": traceId,
            "name": name,
            "endTime": endTime,
            "completionStartTime": completionStartTime,
            "model": model,
            "modelParameters": modelParameters,
            "prompt": prompt,
            "version": version,
            "metadata": metadata,
            "completion": completion,
            "usage": usage,
            "level": level,
            "statusMessage": statusMessage
        }
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.patch(url, json=data) as response:
                return await response.json()

    async def create_score(self, id, traceId, traceIdType, name, value, observationId, comment):
        url = f"{self.base_url}/api/public/scores"
        data = {
            "id": id,
            "traceId": traceId,
            "traceIdType": traceIdType,
            "name": name,
            "value": value,
            "observationId": observationId,
            "comment": comment
        }
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    async def create_span(self, id, traceId, traceIdType, name, startTime, endTime, metadata, input, output, level, statusMessage, parentObservationId, version):
        url = f"{self.base_url}/api/public/spans"
        data = {
            "id": id,
            "traceId": traceId,
            "traceIdType": traceIdType,
            "name": name,
            "startTime": startTime,
            "endTime": endTime,
            "metadata": metadata,
            "input": input,
            "output": output,
            "level": level,
            "statusMessage": statusMessage,
            "parentObservationId": parentObservationId,
            "version": version
        }
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    async def update_span(self, spanId, traceId, endTime, metadata, input, output, level, version, statusMessage):
        url = f"{self.base_url}/api/public/spans"
        data = {
            "spanId": spanId,
            "traceId": traceId,
            "endTime": endTime,
            "metadata": metadata,
            "input": input,
            "output": output,
            "level": level,
            "version": version,
            "statusMessage": statusMessage
        }
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.patch(url, json=data) as response:
                return await response.json()

    async def get_observation(self, observationId):
        url = f"{self.base_url}/api/public/observations/{observationId}"
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.get(url) as response:
                return await response.json()

    async def get_trace(self, traceId):
        url = f"{self.base_url}/api/public/traces/{traceId}"
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.get(url) as response:
                return await response.json()

    async def get_scores(self, page=None, limit=None, userId=None, name=None):
        url = f"{self.base_url}/api/public/scores"
        params = {
            "page": page,
            "limit": limit,
            "userId": userId,
            "name": name
        }
        async with aiohttp.ClientSession(auth=self.auth) as session:
            async with session.get(url, params=params) as response:
                return await response.json()











LANGFUSE_PUBLIC_KEY = "pk-lf-3b6b0cb7-13c8-419b-adc6-b84dc9069021"
LANGFUSE_SECRET_KEY = "sk-lf-2f5aafe7-be17-4d54-af7f-46bb60e58c4c"

client = LangfuseClient(
    base_url="https://cloud.langfuse.com",
    username=LANGFUSE_PUBLIC_KEY,
    password=LANGFUSE_SECRET_KEY
)

import asyncio

loop = asyncio.get_event_loop()

response = loop.run_until_complete(client.create_trace(
    name="Trace 1",
    userId="user-1",
    externalId="external-1",
    release="release-1",
    version="version-1",
    metadata={"key": "value"}
))

print(response)