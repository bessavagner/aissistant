"""Constants for the whole project"""


MODELS = (
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0613',
    'gpt-4',
    'gpt-4-0613',
)
MODELS_EMBEDDING = (
    'text-embedding-ada-002',
)

MAX_TOKENS = (
    (MODELS[0], 4096),
    (MODELS[1], 4096),
    (MODELS[2], 4096),
    (MODELS[3], 4096)
)

ROLES = (  # roles for messages objects
    'system',
    'user',
    'assistant'
)  # see https://platform.openai.com/docs/guides/gpt/chat-completions-api

DEBUG = True
