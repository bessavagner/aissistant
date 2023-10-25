class ContextError(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message


class APIContextError(ContextError):
    def __init__(self, message=''):
        self.message = message
    def __str__(self):
        return self.message
