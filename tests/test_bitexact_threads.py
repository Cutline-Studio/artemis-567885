"""The `bitexact` marker (root conftest.py) pins torch to one thread and restores the count afterwards.
Tests run in file order: the first sets a known count, the marked ones see 1, the next sees the known count again.
Other tests in the suite call torch.set_num_threads themselves, so the count is set here rather than read at import."""
import pytest
import torch

ORIGINAL = torch.get_num_threads()
KNOWN = 3


def test_setup_known_thread_count():
    torch.set_num_threads(KNOWN)
    assert torch.get_num_threads() == KNOWN


@pytest.mark.bitexact
def test_bitexact_marker_runs_the_test_with_one_thread():
    assert torch.get_num_threads() == 1


@pytest.mark.bitexact
@pytest.mark.parametrize("n", [1, 2])
def test_bitexact_marker_applies_to_every_parametrized_case(n):
    assert torch.get_num_threads() == 1


def test_unmarked_test_after_a_bitexact_test_gets_the_previous_thread_count_back():
    try:
        assert torch.get_num_threads() == KNOWN
    finally:
        torch.set_num_threads(ORIGINAL)
