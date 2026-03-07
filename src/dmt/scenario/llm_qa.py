"""LLM question-answering benchmark scenario.

A curated dataset of QA pairs across categories and difficulty levels,
with offline baseline models for comparison without API keys.
"""

import pandas as pd

from dmt.evaluate import Scenario


# ── Scenario descriptor ──────────────────────────────────────────────────────

LLM_QA = Scenario(
    observed_col="expected",
    predicted_col="response",
    entity_col="category",
    merge_on=["question_id"],
    group_by=["category", "difficulty"],
    domain_name="question answering",
    observation_description="curated QA pairs with ground truth",
    entity_description="knowledge categories",
)


# ── Built-in dataset ─────────────────────────────────────────────────────────

_DATASET = [
    # ── Factual: easy ────────────────────────────────────────────────────
    {"question_id": "f01", "question": "What is the capital of France?",
     "expected": "Paris", "category": "factual", "difficulty": "easy"},
    {"question_id": "f02", "question": "What is the capital of Japan?",
     "expected": "Tokyo", "category": "factual", "difficulty": "easy"},
    {"question_id": "f03", "question": "What is the boiling point of water in Celsius?",
     "expected": "100", "category": "factual", "difficulty": "easy"},
    {"question_id": "f04", "question": "What planet is closest to the Sun?",
     "expected": "Mercury", "category": "factual", "difficulty": "easy"},
    {"question_id": "f05", "question": "What is the chemical symbol for gold?",
     "expected": "Au", "category": "factual", "difficulty": "easy"},

    # ── Factual: medium ──────────────────────────────────────────────────
    {"question_id": "f06", "question": "What is the speed of light in metres per second?",
     "expected": "299792458", "category": "factual", "difficulty": "medium"},
    {"question_id": "f07", "question": "Who wrote the Principia Mathematica?",
     "expected": "Isaac Newton", "category": "factual", "difficulty": "medium"},
    {"question_id": "f08", "question": "What is the capital of Mongolia?",
     "expected": "Ulaanbaatar", "category": "factual", "difficulty": "medium"},
    {"question_id": "f09", "question": "What year was the transistor invented?",
     "expected": "1947", "category": "factual", "difficulty": "medium"},
    {"question_id": "f10", "question": "What is the atomic number of carbon?",
     "expected": "6", "category": "factual", "difficulty": "medium"},

    # ── Factual: hard ────────────────────────────────────────────────────
    {"question_id": "f11", "question": "What is the Boltzmann constant in J/K?",
     "expected": "1.380649e-23", "category": "factual", "difficulty": "hard"},
    {"question_id": "f12", "question": "What is the half-life of Carbon-14 in years?",
     "expected": "5730", "category": "factual", "difficulty": "hard"},

    # ── Reasoning: easy ──────────────────────────────────────────────────
    {"question_id": "r01", "question": "What is 2 + 2?",
     "expected": "4", "category": "reasoning", "difficulty": "easy"},
    {"question_id": "r02", "question": "What is 3 + 5?",
     "expected": "8", "category": "reasoning", "difficulty": "easy"},
    {"question_id": "r03", "question": "What is 10 * 10?",
     "expected": "100", "category": "reasoning", "difficulty": "easy"},
    {"question_id": "r04", "question": "What is 100 / 4?",
     "expected": "25", "category": "reasoning", "difficulty": "easy"},
    {"question_id": "r05", "question": "What is 7 - 3?",
     "expected": "4", "category": "reasoning", "difficulty": "easy"},

    # ── Reasoning: medium ────────────────────────────────────────────────
    {"question_id": "r06", "question": "What is the square root of 144?",
     "expected": "12", "category": "reasoning", "difficulty": "medium"},
    {"question_id": "r07", "question": "If a train travels 60 km/h for 2 hours, how far does it go in km?",
     "expected": "120", "category": "reasoning", "difficulty": "medium"},
    {"question_id": "r08", "question": "What is 15% of 200?",
     "expected": "30", "category": "reasoning", "difficulty": "medium"},
    {"question_id": "r09", "question": "How many seconds are in one hour?",
     "expected": "3600", "category": "reasoning", "difficulty": "medium"},
    {"question_id": "r10", "question": "What is 2 to the power of 10?",
     "expected": "1024", "category": "reasoning", "difficulty": "medium"},

    # ── Reasoning: hard ──────────────────────────────────────────────────
    {"question_id": "r11", "question": "What is the derivative of x^3 with respect to x?",
     "expected": "3x^2", "category": "reasoning", "difficulty": "hard"},
    {"question_id": "r12", "question": "What is the integral of 2x dx?",
     "expected": "x^2", "category": "reasoning", "difficulty": "hard"},
    {"question_id": "r13", "question": "What is the sum of the first 100 positive integers?",
     "expected": "5050", "category": "reasoning", "difficulty": "hard"},

    # ── Coding: easy ─────────────────────────────────────────────────────
    {"question_id": "c01", "question": "In Python, what function prints text to the console?",
     "expected": "print", "category": "coding", "difficulty": "easy"},
    {"question_id": "c02", "question": "In Python, what keyword defines a function?",
     "expected": "def", "category": "coding", "difficulty": "easy"},
    {"question_id": "c03", "question": "In Python, what built-in function returns the length of a list?",
     "expected": "len", "category": "coding", "difficulty": "easy"},

    # ── Coding: medium ───────────────────────────────────────────────────
    {"question_id": "c04", "question": "What is the time complexity of binary search?",
     "expected": "O(log n)", "category": "coding", "difficulty": "medium"},
    {"question_id": "c05", "question": "In Python, what module provides regular expressions?",
     "expected": "re", "category": "coding", "difficulty": "medium"},
    {"question_id": "c06", "question": "What data structure uses LIFO ordering?",
     "expected": "stack", "category": "coding", "difficulty": "medium"},

    # ── Coding: hard ─────────────────────────────────────────────────────
    {"question_id": "c07", "question": "What is the amortised time complexity of appending to a Python list?",
     "expected": "O(1)", "category": "coding", "difficulty": "hard"},
    {"question_id": "c08", "question": "In Big-O notation, what is the worst-case time complexity of quicksort?",
     "expected": "O(n^2)", "category": "coding", "difficulty": "hard"},

    # ── Creative: easy ───────────────────────────────────────────────────
    {"question_id": "v01", "question": "Complete the phrase: 'To be or not to ___'",
     "expected": "be", "category": "creative", "difficulty": "easy"},
    {"question_id": "v02", "question": "What colour do you get when you mix red and blue?",
     "expected": "purple", "category": "creative", "difficulty": "easy"},

    # ── Creative: medium ─────────────────────────────────────────────────
    {"question_id": "v03", "question": "What figure of speech is 'the world is a stage'?",
     "expected": "metaphor", "category": "creative", "difficulty": "medium"},
    {"question_id": "v04", "question": "What literary device uses 'like' or 'as' for comparison?",
     "expected": "simile", "category": "creative", "difficulty": "medium"},

    # ── Creative: hard ───────────────────────────────────────────────────
    {"question_id": "v05", "question": "What rhetorical device repeats the same word at the start of successive clauses?",
     "expected": "anaphora", "category": "creative", "difficulty": "hard"},
]


def generate_dataset() -> pd.DataFrame:
    """Return the built-in QA dataset as a DataFrame.

    Columns: question_id, question, expected, category, difficulty
    """
    return pd.DataFrame(_DATASET)
