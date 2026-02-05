from collections import defaultdict
import re
from typing import Union, List

from verl import DataProto


def remove_special_tokens(
    responses: List[str], special_token_list: List[str], special_tokens_to_keep: List[str]
) -> List[str]:
    special_token_set = list(set(special_token_list))
    special_tokens_to_keep = set(special_tokens_to_keep)
    for special_token in special_token_set:
        if special_token not in special_tokens_to_keep:
            responses = [response.replace(special_token, "") for response in responses]
    return responses


def count_tokens_between_ids(
    response_tokens: List[List[int]],
    open_id: int,
    close_id: int,
) -> List[int]:
    """
    Count thinking tokens for a batch of tokenized responses.
    """
    thinking_counts = []
    for tokens in response_tokens:
        thinking_token_count = 0
        i = 0
        while i < len(tokens):
            if tokens[i] == open_id:
                i += 1
                start_pos = i
                while i < len(tokens) and tokens[i] != close_id:
                    i += 1
                if i < len(tokens) and tokens[i] == close_id:
                    thinking_token_count += i - start_pos
                    i += 1
                else:
                    thinking_token_count += len(tokens) - start_pos
                    break
            else:
                i += 1
        thinking_counts.append(thinking_token_count)
    return thinking_counts


def build_reward_dict(data: DataProto, outer_key: str, inner_key: str, value_key: str):
    outer = data.non_tensor_batch[outer_key]
    inner = data.non_tensor_batch[inner_key]
    values = data.non_tensor_batch[value_key]

    reward_dict = defaultdict(dict)
    for o, i, v in zip(outer, inner, values):
        reward_dict[o][i] = v

    return reward_dict


def extract_boxed_expressions(text: str) -> Union[List[str], str]:
    """
    Extracts LaTeX boxed expressions from a string.

    This function uses regex to find all instances of LaTeX boxed expressions
    in the format \\boxed{...} and returns them as a list. If no matches are
    found, it returns a descriptive message.

    Args:
        text: Input string containing LaTeX expressions.

    Returns:
        Union[List[str], str]: A list of extracted boxed expressions if found,
            or the string "No matches" if no boxed expressions are found.

    Example:
        >>> extract_boxed_expressions("The answer is \\boxed{42} and \\boxed{x + y}")
        ['42', 'x + y']
        >>> extract_boxed_expressions("No boxed expressions here")
        'No matches'
    """
    pattern = r"\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}"
    matches = re.findall(pattern, text)
    return matches if matches else "No matches"
