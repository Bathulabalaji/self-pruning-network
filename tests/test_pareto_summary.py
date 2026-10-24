"""The Pareto claim is computed from run records, not typed in by hand."""

from experiments.pareto import summarize


def _row(criterion, target, seed, accuracy, achieved=None, nnz=100, numel=1000):
    if achieved is None:
        achieved = target
    return {
        "criterion": criterion,
        "target_sparsity": target,
        "seed": seed,
        "test_accuracy": accuracy,
        "achieved_sparsity": achieved,
        "nnz": nnz,
        "numel": numel,
    }


def test_claim_reports_means_and_flags_an_overlapping_gap():
    rows = []
    for seed, taylor, magnitude in ((0, 0.97, 0.96), (1, 0.95, 0.97), (2, 0.96, 0.955)):
        rows.append(_row("taylor", 0.9, seed, taylor, nnz=100))
        rows.append(_row("magnitude", 0.9, seed, magnitude, nnz=100))
    summary = summarize(rows)
    assert "0.9600" in summary["claim"]
    assert "magnitude" in summary["claim"]
    assert summary["gap_exceeds_sum_of_seed_stds"] is False
    assert "not distinguishable" in summary["claim"]
    assert summary["points"][0]["sparse_flops_batch_128"] == 2 * 100 * 128


def test_claim_flags_a_gap_larger_than_the_seed_noise():
    rows = []
    for seed in (0, 1, 2):
        rows.append(_row("taylor", 0.9, seed, 0.97))
        rows.append(_row("magnitude", 0.9, seed, 0.90))
    summary = summarize(rows)
    assert summary["gap_exceeds_sum_of_seed_stds"] is True
    assert "larger than the sum" in summary["claim"]
