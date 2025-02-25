import json
from enum import Enum
from typing import Any


class ResponseStatus(Enum):
    """
    Represents the status of a response.

    This enum is used to indicate whether an operation was successful or encountered an error.

    ## Members
    - **SUCCESS ("SUCCESS")**: The operation completed successfully.
    - **ERROR ("ERROR")**: The operation failed due to an error.
    """

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class Response:
    """
    Represents a standardized API response.

    This class encapsulates a response status, an optional message, and optional data.

    ## Attributes
    - **status** (`ResponseStatus`): The status of the response (either SUCCESS or ERROR).
    - **message** (`str`, optional): An optional message providing additional details.
    - **data** (`Any`, optional): Any additional data to be included in the response.
    """

    def __init__(self, status: ResponseStatus, message: str = None, data: Any = None):
        self.status = status
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        """Converts a response to a dictionary format."""
        return {"status": self.status.name, "message": self.message, "data": self.data}

    def to_json(self) -> str:
        """Serializes the response as a JSON string."""
        return json.dumps(self.to_dict())
