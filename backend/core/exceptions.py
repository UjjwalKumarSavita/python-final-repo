"""
Application-specific exceptions to keep error handling consistent.
"""

class AppError(Exception):
    """Base app error."""
    pass

class UnsupportedFileType(AppError):
    """Raised when uploaded file type is not supported."""
    pass

class DocumentNotFound(AppError):
    """Raised when a document id does not exist."""
    pass

class ProcessingError(AppError):
    """Raised for generic processing failures."""
    pass
