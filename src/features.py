import re


def get_prompt_text(row: dict, dataset: str) -> str:
    if dataset == "mbpp":
        return row.get("text", "")
    return row.get("problem", "")


def get_answer_text(row: dict, dataset: str) -> str:
    if dataset == "mbpp":
        return row.get("code", "")
    if dataset == "coding":
        return row.get("answer", "")
    return ""


# --- prompt length features ---

def char_count(text: str) -> int:
    return len(text)

def word_count(text: str) -> int:
    return len(text.split())

def avg_word_length(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


# --- math/latex features ---

def latex_command_count(text: str) -> int:
    return len(re.findall(r'\\[a-zA-Z]+', text))

def fraction_count(text: str) -> int:
    return text.count('\\frac')

def has_integral(text: str) -> int:
    return int('\\int' in text or '∫' in text)

def has_summation(text: str) -> int:
    return int('\\sum' in text or '∑' in text)

def math_env_count(text: str) -> int:
    return len(re.findall(r'\\begin\{', text))


# --- structural features ---

def parenthesis_count(text: str) -> int:
    return text.count('(') + text.count(')') + text.count('{') + text.count('}')

def number_count(text: str) -> int:
    return len(re.findall(r'\b\d+\.?\d*\b', text))

def question_mark_count(text: str) -> int:
    return text.count('?')

def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def arrow_count(text: str) -> int:
    return text.count('-->') + text.count('->') + text.count('→')


# --- coding prompt features ---

def test_case_count(row: dict) -> int:
    return len(row.get("test_list", []))


# --- science/mc features ---

def choice_count(text: str) -> int:
    return len(re.findall(r'\b[A-E][\.\)]\s', text))


# --- coding answer features ---

def answer_char_count(code: str) -> int:
    return len(code)

def answer_line_count(code: str) -> int:
    return len([l for l in code.splitlines() if l.strip()])

def answer_loop_count(code: str) -> int:
    return len(re.findall(r'\bfor\b|\bwhile\b', code))

def answer_function_count(code: str) -> int:
    return len(re.findall(r'\bdef\b', code))

def answer_import_count(code: str) -> int:
    return len(re.findall(r'\bimport\b', code))


# --- master function ---

def extract_features(row: dict, dataset: str) -> dict:
    text = get_prompt_text(row, dataset)
    code = get_answer_text(row, dataset)
    is_coding = dataset in ("coding", "mbpp")

    return {
        # prompt length
        "char_count":            char_count(text),
        "word_count":            word_count(text),
        "avg_word_length":       avg_word_length(text),

        # math/latex
        "latex_command_count":   latex_command_count(text),
        "fraction_count":        fraction_count(text),
        "has_integral":          has_integral(text),
        "has_summation":         has_summation(text),
        "math_env_count":        math_env_count(text),

        # structural
        "parenthesis_count":     parenthesis_count(text),
        "number_count":          number_count(text),
        "question_mark_count":   question_mark_count(text),
        "uppercase_ratio":       uppercase_ratio(text),
        "arrow_count":           arrow_count(text),

        # coding prompt
        "test_case_count":       test_case_count(row),

        # science/mc
        "choice_count":          choice_count(text),

        # coding answer (0 for math/science)
        "answer_char_count":     answer_char_count(code) if is_coding else 0,
        "answer_line_count":     answer_line_count(code) if is_coding else 0,
        "answer_loop_count":     answer_loop_count(code) if is_coding else 0,
        "answer_function_count": answer_function_count(code) if is_coding else 0,
        "answer_import_count":   answer_import_count(code) if is_coding else 0,
    }