import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from bench_classifier import macro_f1, parse_ranked_list, top_k_accuracy


def test_parse_ranked_list_standard():
    text = "1. Pinot Noir\n2. Merlot\n3. Syrah"
    assert parse_ranked_list(text) == ["Pinot Noir", "Merlot", "Syrah"]


def test_parse_ranked_list_fallback_no_numbers():
    text = "Pinot Noir"
    assert parse_ranked_list(text) == ["Pinot Noir"]


def test_parse_ranked_list_strips_markdown_and_explanation_when_classes_given():
    # the exact real bug: Claude wraps the answer in markdown + appends an
    # explanation on the same line -- exact string equality against a bare
    # ground-truth label failed almost every time until this was fixed.
    text = ("1. **Chenin Blanc** — crisp acidity, and the peach/almond/quince-like "
            "notes with a subtle spice are hallmark of medium-bodied Chenin.\n"
            "2. **Viognier** — peach and floral fruit are classic.\n"
            "3. **Riesling** — the freshness and light fruit fit.")
    classes = ["Chenin Blanc", "Viognier", "Riesling", "Pinot Noir"]
    assert parse_ranked_list(text, classes) == ["Chenin Blanc", "Viognier", "Riesling"]


def test_parse_ranked_list_longest_match_wins_on_shared_prefix():
    text = "1. Cabernet Sauvignon is a classic full-bodied red."
    classes = ["Cabernet Franc", "Cabernet Sauvignon", "Merlot"]
    assert parse_ranked_list(text, classes) == ["Cabernet Sauvignon"]


def test_parse_ranked_list_no_class_match_falls_back_to_raw_line():
    text = "1. Some completely unrelated answer"
    classes = ["Chenin Blanc", "Viognier"]
    result = parse_ranked_list(text, classes)
    assert result == ["Some completely unrelated answer"]


def test_top_1_accuracy():
    preds = [["A", "B"], ["B", "A"], ["C", "A"]]
    truths = ["A", "A", "C"]
    assert top_k_accuracy(preds, truths, k=1) == 2 / 3


def test_top_3_accuracy_more_lenient():
    preds = [["A", "B", "C"], ["B", "A", "D"]]
    truths = ["C", "D"]
    assert top_k_accuracy(preds, truths, k=3) == 1.0
    assert top_k_accuracy(preds, truths, k=1) == 0.0


def test_macro_f1_perfect():
    preds = [["A"], ["B"], ["C"]]
    truths = ["A", "B", "C"]
    assert macro_f1(preds, truths) == 1.0


def test_macro_f1_all_wrong():
    preds = [["B"], ["C"], ["A"]]
    truths = ["A", "B", "C"]
    assert macro_f1(preds, truths) == 0.0
