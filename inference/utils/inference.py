from typing import List, Dict, Optional, Union, Tuple

from transformers import PreTrainedTokenizer
import vllm


def run_inference(
    llm: vllm.LLM,
    inputs: Union[str, List[str]],
    sampling_params: vllm.SamplingParams,
) -> List[List[str]]:
    """
    Runs inference using a vLLM language model on provided inputs.

    Args:
        llm (vllm.LLM): vLLM language model.
        inputs (str or list[str]): Input prompts for the model.
        sampling_params (vllm.SamplingParams): Parameters for controlling sampling behavior.

    Returns:
        list[list[str]]: A list of predictions, where each inner list contains multiple output strings.
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    print(f"Sampling params: {sampling_params}")

    raw_output = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    predictions_store = [[s.text for s in out.outputs] for out in raw_output]
    return predictions_store


def create_messages(
    user_inputs: Union[str, List[str]],
    assistant_inputs: Optional[Union[str, List[str]]] = None,
    system: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Creates a list of chat messages formatted for model inference.
    Expects user inputs and optional assistant inputs to be strings or lists of strings.
    If provided as a list, user inputs should be of equal length or one element longer than assistant inputs.

    Args:
        user_inputs (str or list[str]): The user query or queries to be answered.
        assistant_inputs (str or list[str], optional): The assistant response or responses.
        system (str, optional): Optional system message for context.

    Returns:
        list[dict]: List of messages.
    """
    if isinstance(user_inputs, str):
        user_inputs = [user_inputs]
    if isinstance(assistant_inputs, str):
        assistant_inputs = [assistant_inputs]

    if system:
        messages = [{"role": "system", "content": system}]
    else:
        messages = []

    num_user = len(user_inputs)
    num_assistant = len(assistant_inputs) if assistant_inputs else 0

    if num_user < num_assistant or num_user > num_assistant + 1:
        raise ValueError(
            "The number of user inputs must be equal to or one greater than the number of assistant inputs."
        )

    for i in range(num_user):
        messages.append({"role": "user", "content": user_inputs[i]})
        if assistant_inputs and i < num_assistant:
            messages.append({"role": "assistant", "content": assistant_inputs[i]})

    return messages


def render_chat_to_prompt(messages, tokenizer, enable_thinking=False, add_generation_prompt=True):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking
    )


def remove_special_tokens(
    responses: List[str], special_token_list: List[str], special_tokens_to_keep: List[str]
) -> List[str]:
    """
    Removes special tokens from responses.
    """
    special_token_set = list(set(special_token_list))
    special_tokens_to_keep = set(special_tokens_to_keep)
    for special_token in special_token_set:
        if special_token not in special_tokens_to_keep:
            responses = [response.replace(special_token, "") for response in responses]
    return responses
