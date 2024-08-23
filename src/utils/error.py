class AdsorbatesNotTaggedError(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.args[0]
    
class SurfaceNotTaggedError(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.args[0]
    
class BulkTagError(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.args[0]

class TooManyAdsorbatesError(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.args[0]

class NoStructureMatchQueryError(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.args[0]

class StructuresNotValidatedError(Exception):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.args[0]
