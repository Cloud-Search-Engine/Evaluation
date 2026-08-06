from metrics import mrr, precision_at_k, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["x", "a", "y"], 3) == 0.5


def test_precision_at_k():
    assert precision_at_k(["a"], ["a", "b"], 2) == 0.5


def test_mrr():
    assert mrr(["b"], ["a", "b", "c"]) == 0.5
