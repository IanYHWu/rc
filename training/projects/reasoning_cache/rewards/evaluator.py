from typing import List, Union, Dict
import re

from math_verify import parse, verify


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
    def __init__(self, add_boxed_to_gold_answer: bool = True, extract_candidate_answer: bool = True):
        self.add_boxed_to_gold_answer = add_boxed_to_gold_answer
        self.extract_candidate_answer = extract_candidate_answer

    @staticmethod
    def add_boxed(text):
        return f"\\boxed{{{text}}}"

    def __call__(self, candidate_answer: str, gold_answer: str) -> bool:
        return self.evaluate_answer(candidate_answer, gold_answer)

    def evaluate_answer(self, candidate_answer: str, gold_answer: str) -> bool:
        if self.add_boxed_to_gold_answer:
            gold_answer = self.add_boxed(gold_answer)
        if self.extract_candidate_answer:
            candidate_answer = extract_boxed_expressions(candidate_answer)
        if candidate_answer is None:
            return {"result": False, "parsed": None}

        parsed_candidate_answer = parse(candidate_answer)
        parsed_gold_answer = parse(gold_answer)
        evaluation_result = verify(parsed_candidate_answer, parsed_gold_answer)
        if isinstance(parsed_candidate_answer, list) and len(parsed_candidate_answer) > 1:
            return {"result": evaluation_result, "parsed": parsed_candidate_answer[1]}
        else:
            return {"result": evaluation_result, "parsed": None}
        

def compute_score(candidate_answer: str, gold_answer: str) -> float:
    evaluator = MathEvaluator()
    result = evaluator.evaluate_answer(candidate_answer, gold_answer)
    return float(result["result"])


def compute_score_verl(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict[str, str]):
    evaluator = MathEvaluator()
    result = evaluator.evaluate_answer(solution_str, ground_truth)
    return {
        "score": float(result["result"]),
    }