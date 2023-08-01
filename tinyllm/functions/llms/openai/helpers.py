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
