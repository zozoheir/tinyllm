from langchain.callbacks.openai_info import standardize_model_name, MODEL_COST_PER_1K_TOKENS, \
    get_openai_token_cost_for_model


def get_user_message(message):
    return {'role': 'user',
            'content': message}

def get_system_message(message):
    return {'role': 'system',
            'content': message}

def get_function_message(content, name):
    return {'role': 'function',
            'name': name,
            'content': content}

def get_assistant_message(content):
    return {'role': 'assistant',
            'content': content}


def get_openai_api_cost(api_response):
    """Collect token usage."""
    token_usage = api_response["usage"]
    completion_tokens = token_usage.get("completion_tokens", 0)
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    total_tokens = token_usage.get("total_tokens", 0)
    model_name = standardize_model_name(api_response['model'])
    total_cost = 0
    if model_name in MODEL_COST_PER_1K_TOKENS:
        completion_cost = get_openai_token_cost_for_model(
            model_name, completion_tokens, is_completion=True
        )
        prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
        total_cost += prompt_cost + completion_cost
    return {
        'cost': total_cost,
        'total_tokens': total_tokens,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
    }
