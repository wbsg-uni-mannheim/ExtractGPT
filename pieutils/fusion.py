from collections import Counter
from typing import List

from pydantic import BaseModel


def majority_vote(strings: List[str]) -> str:
    """
    Determine the majority value in the strings.
    In case of a tie, return the shortest string.
    """
    # Treat 'n/a' as None
    strings = [s if s != 'n/a' else None for s in strings]

    # Filter out None values and non-string values
    valid_strings = [s for s in strings if s is not None and type(s) == str]

    if not valid_strings:
        return 'n/a'

    count = Counter(valid_strings)
    max_count = max(count.values())

    # Get all items that have the max count
    candidates = [key for key, value in count.items() if value == max_count]

    # If only one candidate, return it
    if len(candidates) == 1:
        return candidates[0]

    # Otherwise, return the shortest string among candidates
    return min(candidates, key=len)


def first_occurrence(strings: List[str]) -> str:
    """
    Return the first occurrence of a valid string in the list.
    """
    # Treat 'n/a' as None
    strings = [s if s != 'n/a' else None for s in strings]

    # Filter out None values and non-string values
    valid_strings = [s for s in strings if s is not None and type(s) == str]

    if not valid_strings:
        return 'n/a'

    # Return the first valid string
    return valid_strings[0]


def fuse_models(pydantic_model, *models) -> BaseModel:
    """
    Fuse multiple pydantic models into one.
    """
    fused_data = {}

    for field in pydantic_model.__fields__:
        values = [getattr(model, field) for model in models if model is not None and hasattr(model, field)]
        fused_data[field] = first_occurrence(values)

    return pydantic_model(**fused_data)
