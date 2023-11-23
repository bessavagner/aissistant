"""
    context_chat.agents.constants

Constants for module agents. Some are used in other modules.
Most constants comes from OpenAI documentation. See:
https://platform.openai.com/docs/models
"""

MODELS = (
    'gpt-3.5-turbo-1106',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-4',
    'gpt-4-32k',
    'gpt-4-0613',
    'gpt-4-1106-preview'
)

MAX_TOKENS = (
    (MODELS[0], 16385),
    (MODELS[1], 4096),
    (MODELS[2], 16385),
    (MODELS[3], 8192),
    (MODELS[4], 32768),
    (MODELS[5], 8192),
    (MODELS[6], 128000)
)

ROLES = (  # roles for messages objects
    'system',
    'user',
    'assistant'
)  # see https://platform.openai.com/docs/guides/gpt/chat-completions-api


DEBUG = False

"""Constants for embeddings operations"""

MODELS_EMBEDDING = (
    'text-embedding-ada-002',
)

TOKENIZER = (
    'cl100k_base',
)

EMBEDDINGS_COLUMNS = ('chunks', 'n_tokens', 'embeddings',)
