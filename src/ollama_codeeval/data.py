"""Loads and processes human eval data from JSONL files. Handles gzipped files and includes notes about data quality improvements."""

import gzip
import json
import os
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Iterator

DATA_URL = "https://github.com/marcusm117/human-eval-enhanced/blob/main/data/human-eval-enhanced-202307.jsonl.gz"
DATA_FILENAME = "human-eval-enhanced-202307.jsonl.gz"


class HumanEvalTask(Enum):
    """Entry points for the HumanEval benchmark tasks."""

    HAS_CLOSE_ELEMENTS = "has_close_elements"
    SEPARATE_PAREN_GROUPS = "separate_paren_groups"
    TRUNCATE_NUMBER = "truncate_number"
    BELOW_ZERO = "below_zero"
    MEAN_ABSOLUTE_DEVIATION = "mean_absolute_deviation"
    INTERSPERSE = "intersperse"
    PARSE_NESTED_PARENS = "parse_nested_parens"
    FILTER_BY_SUBSTRING = "filter_by_substring"
    SUM_PRODUCT = "sum_product"
    ROLLING_MAX = "rolling_max"
    MAKE_PALINDROME = "make_palindrome"
    STRING_XOR = "string_xor"
    LONGEST = "longest"
    GREATEST_COMMON_DIVISOR = "greatest_common_divisor"
    ALL_PREFIXES = "all_prefixes"
    STRING_SEQUENCE = "string_sequence"
    COUNT_DISTINCT_CHARACTERS = "count_distinct_characters"
    PARSE_MUSIC = "parse_music"
    HOW_MANY_TIMES = "how_many_times"
    SORT_NUMBERS = "sort_numbers"
    FIND_CLOSEST_ELEMENTS = "find_closest_elements"
    RESCALE_TO_UNIT = "rescale_to_unit"
    FILTER_INTEGERS = "filter_integers"
    STRLEN = "strlen"
    LARGEST_DIVISOR = "largest_divisor"
    FACTORIZE = "factorize"
    REMOVE_DUPLICATES = "remove_duplicates"
    FLIP_CASE = "flip_case"
    CONCATENATE = "concatenate"
    FILTER_BY_PREFIX = "filter_by_prefix"
    GET_POSITIVE = "get_positive"
    IS_PRIME = "is_prime"
    FIND_ZERO = "find_zero"
    SORT_THIRD = "sort_third"
    UNIQUE = "unique"
    MAX_ELEMENT = "max_element"
    FIZZ_BUZZ = "fizz_buzz"
    SORT_EVEN = "sort_even"
    DECODE_CYCLIC = "decode_cyclic"
    PRIME_FIB = "prime_fib"
    TRIPLES_SUM_TO_ZERO = "triples_sum_to_zero"
    CAR_RACE_COLLISION = "car_race_collision"
    INCR_LIST = "incr_list"
    PAIRS_SUM_TO_ZERO = "pairs_sum_to_zero"
    CHANGE_BASE = "change_base"
    TRIANGLE_AREA = "triangle_area"
    FIB4 = "fib4"
    MEDIAN = "median"
    IS_PALINDROME = "is_palindrome"
    MODP = "modp"
    DECODE_SHIFT = "decode_shift"
    REMOVE_VOWELS = "remove_vowels"
    BELOW_THRESHOLD = "below_threshold"
    ADD = "add"
    SAME_CHARS = "same_chars"
    FIB = "fib"
    CORRECT_BRACKETING = "correct_bracketing"
    MONOTONIC = "monotonic"
    COMMON = "common"
    LARGEST_PRIME_FACTOR = "largest_prime_factor"
    SUM_TO_N = "sum_to_n"
    DERIVATIVE = "derivative"
    FIBFIB = "fibfib"
    VOWELS_COUNT = "vowels_count"
    CIRCULAR_SHIFT = "circular_shift"
    DIGITSUM = "digitSum"
    FRUIT_DISTRIBUTION = "fruit_distribution"
    PLUCK = "pluck"
    SEARCH = "search"
    STRANGE_SORT_LIST = "strange_sort_list"
    WILL_IT_FLY = "will_it_fly"
    SMALLEST_CHANGE = "smallest_change"
    TOTAL_MATCH = "total_match"
    IS_MULTIPLY_PRIME = "is_multiply_prime"
    IS_SIMPLE_POWER = "is_simple_power"
    IS_CUBE = "iscube"
    HEX_KEY = "hex_key"
    DECIMAL_TO_BINARY = "decimal_to_binary"
    IS_HAPPY = "is_happy"
    NUMERICAL_LETTER_GRADE = "numerical_letter_grade"
    PRIME_LENGTH = "prime_length"
    STARTS_ONE_ENDS = "starts_one_ends"
    SOLVE = "solve"
    ANTI_SHUFFLE = "anti_shuffle"
    GET_ROW = "get_row"
    SORT_ARRAY = "sort_array"
    ENCRYPT = "encrypt"
    NEXT_SMALLEST = "next_smallest"
    IS_BORED = "is_bored"
    ANY_INT = "any_int"
    ENCODE = "encode"
    SKJKASDKD = "skjkasdkd"
    CHECK_DICT_CASE = "check_dict_case"
    COUNT_UP_TO = "count_up_to"
    MULTIPLY = "multiply"
    COUNT_UPPER = "count_upper"
    CLOSEST_INTEGER = "closest_integer"
    MAKE_A_PILE = "make_a_pile"
    WORDS_STRING = "words_string"
    CHOOSE_NUM = "choose_num"
    ROUNDED_AVG = "rounded_avg"
    UNIQUE_DIGITS = "unique_digits"
    BY_LENGTH = "by_length"
    EVEN_ODD_PALINDROME = "even_odd_palindrome"
    COUNT_NUMS = "count_nums"
    MOVE_ONE_BALL = "move_one_ball"
    EXCHANGE = "exchange"
    HISTOGRAM = "histogram"
    REVERSE_DELETE = "reverse_delete"
    ODD_COUNT = "odd_count"
    MINSUBARRAYSUM = "minSubArraySum"
    MAX_FILL = "max_fill"
    SELECT_WORDS = "select_words"
    GET_CLOSEST_VOWEL = "get_closest_vowel"
    MATCH_PARENS = "match_parens"
    MAXIMUM = "maximum"
    SOLUTION = "solution"
    ADD_ELEMENTS = "add_elements"
    GET_ODD_COLLATZ = "get_odd_collatz"
    VALID_DATE = "valid_date"
    SPLIT_WORDS = "split_words"
    IS_SORTED = "is_sorted"
    INTERSECTION = "intersection"
    PROD_SIGNS = "prod_signs"
    MINPATH = "minPath"
    TRI = "tri"
    DIGITS = "digits"
    IS_NESTED = "is_nested"
    SUM_SQUARES = "sum_squares"
    CHECK_IF_LAST_CHAR_IS_A_LETTER = "check_if_last_char_is_a_letter"
    CAN_ARRANGE = "can_arrange"
    LARGEST_SMALLEST_INTEGERS = "largest_smallest_integers"
    COMPARE_ONE = "compare_one"
    IS_EQUAL_TO_SUM_EVEN = "is_equal_to_sum_even"
    SPECIAL_FACTORIAL = "special_factorial"
    FIX_SPACES = "fix_spaces"
    FILE_NAME_CHECK = "file_name_check"
    WORDS_IN_SENTENCE = "words_in_sentence"
    SIMPLIFY = "simplify"
    ORDER_BY_POINTS = "order_by_points"
    SPECIALFILTER = "specialFilter"
    GET_MAX_TRIPLES = "get_max_triples"
    BF = "bf"
    SORTED_LIST_SUM = "sorted_list_sum"
    X_OR_Y = "x_or_y"
    DOUBLE_THE_DIFFERENCE = "double_the_difference"
    COMPARE = "compare"
    STRONGEST_EXTENSION = "Strongest_Extension"
    CYCPATTERN_CHECK = "cycpattern_check"
    EVEN_ODD_COUNT = "even_odd_count"
    INT_TO_MINI_ROMAN = "int_to_mini_roman"
    RIGHT_ANGLE_TRIANGLE = "right_angle_triangle"
    FIND_MAX = "find_max"
    EAT = "eat"
    DO_ALGEBRA = "do_algebra"
    STRING_TO_MD5 = "string_to_md5"
    GENERATE_INTEGERS = "generate_integers"


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _get_cache_dir() -> Path:
    """Return the platform-appropriate cache directory for ollama-codeeval."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = base / "ollama-codeeval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_data_path() -> Path:
    """Return the path to the HumanEval dataset, downloading it if necessary.

    Looks in three locations in order:
    1. Local development path (data/ directory in repo root)
    2. Cache directory (downloaded from GitHub)
    """
    # Check local development path first
    local_path = Path(__file__).parent.parent.parent / "data" / DATA_FILENAME
    if local_path.exists():
        return local_path

    # Check/download to cache
    cache_path = _get_cache_dir() / DATA_FILENAME
    if not cache_path.exists():
        print(f"Downloading HumanEval dataset to {cache_path}...")
        urllib.request.urlretrieve(DATA_URL, cache_path)  # noqa: B310 # nosec: B310
        print("Download complete.")
    return cache_path


def load_jsonl_gz(file_path: str | Path) -> Iterator[dict]:
    """
    Loads a gzipped JSONL file line by line.

    Args:
        file_path: The path to the .jsonl.gz file.

    Yields:
        A dictionary for each JSON object in the file.
    """
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_data(
    filter: list[HumanEvalTask] | None = None,
    path: Path | None = None,
) -> list[dict[str, str]]:
    """Load a HumanEval dataset from local JSONL.

    Uses the fixed data from https://github.com/openai/human-eval/pull/23
    (thanks marcusm117) rather than the error-prone original.
    """
    data = list(load_jsonl_gz(path or _get_data_path()))
    if filter:
        keys = {f.value for f in filter}
        data = [r for r in data if r["entry_point"] in keys]
    return data
