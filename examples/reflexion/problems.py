"""Coding problems for Reflexion experiment.

Each problem has:
- Description and function signature
- Test cases for validation
"""

from dataclasses import dataclass
from typing import List, Any, Callable, Tuple


@dataclass
class CodingProblem:
    """A coding problem with test cases."""
    name: str
    description: str
    function_name: str
    function_signature: str
    test_cases: List[Tuple[Tuple, Any]]  # ((args...), expected_output)
    difficulty: str  # "easy", "medium", "hard"
    
    def validate(self, func: Callable) -> Tuple[float, int, int]:
        """Validate a function against test cases.
        
        Returns:
            Tuple of (accuracy, correct_count, total_count)
        """
        correct = 0
        total = len(self.test_cases)
        
        for args, expected in self.test_cases:
            try:
                if isinstance(args, tuple):
                    result = func(*args)
                else:
                    result = func(args)
                if result == expected:
                    correct += 1
            except Exception:
                pass
        
        return correct / total if total > 0 else 0.0, correct, total
    
    def get_prompt(self) -> str:
        """Get the prompt for code generation."""
        return f"""{self.description}

Function signature: {self.function_signature}

Write only the Python function code, no explanations."""


# ============================================================================
# PROBLEM COLLECTION
# ============================================================================

PROBLEMS = [
    # Easy problems
    CodingProblem(
        name="two_sum",
        description="Given an array of integers nums and an integer target, return indices of the two numbers that add up to target. Assume exactly one solution exists.",
        function_name="two_sum",
        function_signature="def two_sum(nums: list, target: int) -> list:",
        test_cases=[
            (([2, 7, 11, 15], 9), [0, 1]),
            (([3, 2, 4], 6), [1, 2]),
            (([3, 3], 6), [0, 1]),
        ],
        difficulty="easy",
    ),
    
    CodingProblem(
        name="fizzbuzz",
        description="Return 'FizzBuzz' if n is divisible by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible by 5, otherwise return the number as a string.",
        function_name="fizzbuzz",
        function_signature="def fizzbuzz(n: int) -> str:",
        test_cases=[
            ((15,), "FizzBuzz"),
            ((3,), "Fizz"),
            ((5,), "Buzz"),
            ((7,), "7"),
            ((30,), "FizzBuzz"),
        ],
        difficulty="easy",
    ),
    
    CodingProblem(
        name="is_palindrome",
        description="Return True if the string is a palindrome (ignoring case and non-alphanumeric characters), False otherwise.",
        function_name="is_palindrome",
        function_signature="def is_palindrome(s: str) -> bool:",
        test_cases=[
            (("A man a plan a canal Panama",), True),
            (("race a car",), False),
            (("",), True),
            (("a",), True),
        ],
        difficulty="easy",
    ),
    
    # Medium problems
    CodingProblem(
        name="valid_parentheses",
        description="Given a string containing just '(', ')', '{', '}', '[' and ']', determine if the input string is valid. Brackets must close in correct order.",
        function_name="is_valid",
        function_signature="def is_valid(s: str) -> bool:",
        test_cases=[
            (("()",), True),
            (("()[]{}",), True),
            (("(]",), False),
            (("([)]",), False),
            (("{[]}",), True),
        ],
        difficulty="medium",
    ),
    
    CodingProblem(
        name="reverse_words",
        description="Given a string, reverse the order of words. Words are separated by spaces.",
        function_name="reverse_words",
        function_signature="def reverse_words(s: str) -> str:",
        test_cases=[
            (("the sky is blue",), "blue is sky the"),
            (("  hello world  ",), "world hello"),
            (("a",), "a"),
        ],
        difficulty="medium",
    ),
    
    # Hard problems
    CodingProblem(
        name="longest_substring",
        description="Find the length of the longest substring without repeating characters.",
        function_name="length_of_longest_substring",
        function_signature="def length_of_longest_substring(s: str) -> int:",
        test_cases=[
            (("abcabcbb",), 3),
            (("bbbbb",), 1),
            (("pwwkew",), 3),
            (("",), 0),
        ],
        difficulty="hard",
    ),
]


def get_problems(difficulty: str = None) -> List[CodingProblem]:
    """Get problems, optionally filtered by difficulty."""
    if difficulty:
        return [p for p in PROBLEMS if p.difficulty == difficulty]
    return PROBLEMS


def get_problem_by_name(name: str) -> CodingProblem:
    """Get a specific problem by name."""
    for problem in PROBLEMS:
        if problem.name == name:
            return problem
    raise ValueError(f"Problem '{name}' not found")


if __name__ == "__main__":
    print("Available Reflexion Problems:")
    print("-" * 50)
    for problem in PROBLEMS:
        print(f"\n{problem.name} [{problem.difficulty}]")
        print(f"  {problem.description[:60]}...")
        print(f"  Tests: {len(problem.test_cases)} cases")
