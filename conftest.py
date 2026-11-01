"""Shared pytest setup for tests/ and checks/.

`@pytest.mark.bitexact` pins torch to one intra-op thread for that test. Bit-exact
(torch.equal) comparisons between forward passes on different inputs or batch shapes
can differ in the last bit when a multi-threaded reduction splits work differently
(seen with 32 threads on torch 2.5.1, 2026-10-26). One thread makes the summation
order fixed, so a difference means a real leak, not reduction order."""
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "bitexact: run with torch.set_num_threads(1) so torch.equal checks do not depend on thread count")


@pytest.fixture(autouse=True)
def _single_thread_for_bitexact(request):
    if request.node.get_closest_marker("bitexact") is None:
        yield
        return
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)
