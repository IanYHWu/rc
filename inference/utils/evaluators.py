from typing import List, Union
import re
import warnings
import logging

from math_verify import verify, parse

warnings.filterwarnings("ignore", message=".*Timeout is disabled.*")
logging.getLogger("math_verify").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def extract_boxed_expressions(text: str) -> Union[List[str], str]:
    """
    Extracts LaTeX boxed expressions from a string.

    Args:
        text (str): Input string containing LaTeX expressions.

    Returns:
        list[str] or str: Extracted expressions or a message if none found.
    """
    pattern = r"\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


class MathEvaluator:
    def __init__(self, add_boxed_to_gold_answer: bool = True, enforce_boxed: bool = True):
        self.add_boxed_to_gold_answer = add_boxed_to_gold_answer
        self.enforce_boxed = enforce_boxed

    @staticmethod
    def add_boxed(text):
        return f"\\boxed{{{text}}}"

    def __call__(self, candidate_answer: str, gold_answer: str) -> bool:
        return self.evaluate_answer(candidate_answer, gold_answer)

    def evaluate_answer(self, candidate_answer: str, gold_answer: str) -> bool:
        if candidate_answer is None:
            return {"result": False, "parsed": None}
        if self.enforce_boxed:
            if "boxed" not in candidate_answer:
                return {"result": False, "parsed": None}
        if self.add_boxed_to_gold_answer:
            gold_answer = self.add_boxed(gold_answer)
        parsed_candidate_answer = parse(candidate_answer, parsing_timeout=None)
        parsed_gold_answer = parse(gold_answer, parsing_timeout=None)
        evaluation_result = verify(parsed_candidate_answer, parsed_gold_answer, timeout_seconds=None)
        if isinstance(parsed_candidate_answer, list) and len(parsed_candidate_answer) > 1:
            return {"result": evaluation_result, "parsed": parsed_candidate_answer[1]}
        else:
            return {"result": evaluation_result, "parsed": None}

    def parse_answer(self, answer: str) -> str:
        if answer is None:
            return {"parsed": None}
        if self.enforce_boxed:
            if "boxed" not in answer:
                return {"parsed": None}
        parsed_answer = parse(answer, parsing_timeout=None)
        if isinstance(parsed_answer, list) and len(parsed_answer) > 1:
            return {"parsed": parsed_answer[1]}
        else:
            return {"parsed": None}


def evaluate_single(boxed_answer, label):
    """Worker function for multiprocessing."""
    evaluator = MathEvaluator()
    return evaluator(boxed_answer, label)["result"]


def parse_single(answer):
    evaluator = MathEvaluator()
    return evaluator.parse_answer(answer)["parsed"]
