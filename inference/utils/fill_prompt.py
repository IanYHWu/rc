import string
from typing import Dict, Any, List
from itertools import product


def extract_nested_values(data: Dict[str, Any], path: List[str]) -> List[Any]:
    """
    Extract all values from a nested field path.

    Args:
        data: The data dictionary
        path: List representing the path to the nested field (e.g., ['guesses'] for data['guesses'])

    Returns:
        List of all values found at the nested path
    """
    current = data
    for key in path[:-1]:
        if key not in current:
            return []
        current = current[key]

    final_key = path[-1]
    if final_key not in current:
        return []

    nested_data = current[final_key]
    if isinstance(nested_data, dict):
        return list(nested_data.values())
    elif isinstance(nested_data, list):
        return nested_data
    else:
        return [nested_data]


def detect_nested_fields(prompt: str, sample: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Detect which fields in the prompt correspond to nested data structures.

    Args:
        prompt: The format string
        sample: Sample data to check

    Returns:
        Dictionary mapping field names to their values (list for nested, single value for non-nested)
    """
    formatter = string.Formatter()
    field_names = {field_name for _, field_name, _, _ in formatter.parse(prompt) if field_name}

    field_values = {}

    for field_name in field_names:
        # Check if field contains dot notation for nested access
        if "." in field_name:
            path_parts = field_name.split(".")
            values = extract_nested_values(sample, path_parts)
            field_values[field_name] = values
        else:
            # Check if the field itself is a nested structure
            if field_name in sample:
                field_data = sample[field_name]
                if isinstance(field_data, dict):
                    field_values[field_name] = list(field_data.values())
                elif isinstance(field_data, list):
                    field_values[field_name] = field_data
                else:
                    field_values[field_name] = [field_data]
            else:
                field_values[field_name] = [None]

    return field_values


def fill_prompt(sample: Dict[str, Any], prompt: str) -> List[str]:
    """
    Fill prompt with support for nested fields. Handles arbitrary nesting depths.

    Args:
        sample: Sample data
        prompt: Prompt template
        required_fields: Set of required field names (optional)

    Returns:
        List of filled prompts (one for each combination of nested field values)
    """
    required_fields = extract_format_fields(prompt)
    if required_fields:
        check_format_fields(prompt, required_fields)
    else:
        return [prompt]

    field_values = detect_nested_fields(prompt, sample)

    # Check if any field has multiple values (indicating nesting)
    has_nested = any(len(values) > 1 for values in field_values.values())

    if not has_nested:
        # No nested fields, return single prompt
        format_dict = {field: values[0] if values else None for field, values in field_values.items()}
        return prompt.format(**format_dict)

    # Handle nested fields by creating all combinations
    field_names = list(field_values.keys())
    value_lists = list(field_values.values())

    filled_prompts = []
    for value_combination in product(*value_lists):
        format_dict = dict(zip(field_names, value_combination))
        filled_prompts.append(prompt.format(**format_dict))

    return filled_prompts


def check_format_fields(prompt: str, required_fields: set[str]) -> None:
    """
    Check that all required fields are present in the format string.

    Args:
        prompt (str): The format string to check.
        required_fields (set[str]): Set of required field names.

    Raises:
        ValueError: If any required fields are missing from the prompt.
    """
    formatter = string.Formatter()
    found_fields = {field_name for _, field_name, _, _ in formatter.parse(prompt) if field_name}

    missing_fields = required_fields - found_fields
    if missing_fields:
        raise ValueError(f"Prompt is missing required fields: {', '.join(missing_fields)}")


def extract_format_fields(prompt: str) -> set[str]:
    """
    Extract all format field names from a prompt string automatically.

    This function parses a format string and returns all the field names
    that are used in curly braces (e.g., {field_name}).

    Args:
        prompt (str): The format string to analyze (e.g., "Hello {name}, you are {age} years old")

    Returns:
        Set[str]: Set of field names found in the prompt (e.g., {"name", "age"})
    """
    formatter = string.Formatter()
    return {field_name for _, field_name, _, _ in formatter.parse(prompt) if field_name}
