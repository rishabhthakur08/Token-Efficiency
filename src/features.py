import re


def get_prompt_text(row: dict, dataset: str) -> str:
    if dataset == "mbpp":
        return row.get("text", "")
    return row.get("problem", "")


# --- length features ---

def char_count(text: str) -> int:
    """total character length — best single proxy for problem complexity"""
    return len(text)

def word_count(text: str) -> int:
    """word count — independent signal from char count for linguistic density"""
    return len(text.split())

def avg_word_length(text: str) -> float:
    """longer words = more technical vocabulary (strong signal for science)"""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


# --- math/latex features ---

def latex_command_count(text: str) -> int:
    """count of \\command patterns — overall LaTeX complexity for math"""
    return len(re.findall(r'\\[a-zA-Z]+', text))

def fraction_count(text: str) -> int:
    """\\frac specifically signals multi-step algebraic reasoning"""
    return text.count('\\frac')

def has_integral(text: str) -> int:
    """integrals are among the most token-heavy math operations"""
    return int('\\int' in text or '∫' in text)

def has_summation(text: str) -> int:
    """summations require index reasoning distinct from integrals"""
    return int('\\sum' in text or '∑' in text)

def math_env_count(text: str) -> int:
    """count of \\begin{...} environments — signals structured math blocks"""
    return len(re.findall(r'\\begin\{', text))


# --- structural/complexity features ---

def parenthesis_count(text: str) -> int:
    """nested expressions in math and chemistry notation"""
    return text.count('(') + text.count(')') + text.count('{') + text.count('}')

def number_count(text: str) -> int:
    """count of numerical values — meaningful across all domains"""
    return len(re.findall(r'\b\d+\.?\d*\b', text))

def question_mark_count(text: str) -> int:
    """multiple sub-questions force model to address multiple things"""
    return text.count('?')

def uppercase_ratio(text: str) -> float:
    """high uppercase = molecular formulas, acronyms, technical terms (science)"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def arrow_count(text: str) -> int:
    """reaction arrows in chemistry (-->, ->, →) signal multi-step problems"""
    return text.count('-->') + text.count('->') + text.count('→')


# --- coding features ---

def test_case_count(row: dict) -> int:
    """more test cases = more complex function required"""
    return len(row.get("test_list", []))


# --- science/mc features ---

def choice_count(text: str) -> int:
    """number of answer choices — constrains answer space"""
    return len(re.findall(r'\b[A-E][\.\)]\s', text))


# --- master function ---

def extract_features(row: dict, dataset: str) -> dict:
    text = get_prompt_text(row, dataset)

    return {
        # length
        "char_count":           char_count(text),
        "word_count":           word_count(text),
        "avg_word_length":      avg_word_length(text),

        # math/latex
        "latex_command_count":  latex_command_count(text),
        "fraction_count":       fraction_count(text),
        "has_integral":         has_integral(text),
        "has_summation":        has_summation(text),
        "math_env_count":       math_env_count(text),

        # structural
        "parenthesis_count":    parenthesis_count(text),
        "number_count":         number_count(text),
        "question_mark_count":  question_mark_count(text),
        "uppercase_ratio":      uppercase_ratio(text),
        "arrow_count":          arrow_count(text),

        # coding
        "test_case_count":      test_case_count(row),

        # science/mc
        "choice_count":         choice_count(text),
    }
