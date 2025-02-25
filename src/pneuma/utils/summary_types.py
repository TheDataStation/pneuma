from enum import Enum


class SummaryType(Enum):
    """
    Represents the type of a table summary in Pneuma.

    This enum defines the possible summary types.

    ## Members
    - **COLUMN_NARRATION (1)**: LLM narrations for table columns.
    - **ROW_SAMPLE (2)**: Samples of the table rows.
    """
    COLUMN_NARRATION = 1
    ROW_SAMPLE = 2
