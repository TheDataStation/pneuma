from enum import Enum


class TableStatus(Enum):
    """
    Represents the status of a table in the database of Pneuma.

    This enum defines possible states a table can be in during its lifecycle.

    ## Members
    - **REGISTERED (1)**: The table has been registered but not yet summarized.
    - **SUMMARIZED (2)**: The table has been summarized.
    - **DELETED (3)**: The table has been flagged for removal from the DB.
    """
    REGISTERED = 1
    SUMMARIZED = 2
    DELETED = 3
