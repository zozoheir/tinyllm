def get_user_message(message):
    return {'role': 'user',
            'content': message}

def get_system_message(message):
    return {'role': 'system',
            'content': message}