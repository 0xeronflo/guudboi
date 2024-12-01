import json
from datetime import datetime


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for serializing datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def log_header(message):
    """Log a clear, highlighted section header."""
    print(f"\n{'#' * (len(message) + 10)}")
    print(f"### {message} ###")
    print(f"{'#' * (len(message) + 10)}\n")
