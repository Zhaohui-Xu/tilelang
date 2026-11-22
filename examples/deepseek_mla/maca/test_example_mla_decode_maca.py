"""Small correctness regressions for the C500-specific MLA implementation."""

import pytest
import torch
import tilelang.testing


@pytest.mark.parametrize(
    "heads,length,head_tile,splits",
    [
        pytest.param(16, 31, 16, 1, id="shared-query-tail-no-merge"),
        pytest.param(16, 65, 16, 2, id="shared-query-uneven-splits"),
        pytest.param(32, 65, 32, 2, id="fragment-query-two-wave-pairs"),
        pytest.param(64, 65, 64, 2, id="fragment-query-four-wave-pairs"),
    ],
)
@tilelang.testing.requires_cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or "c500" not in torch.cuda.get_device_name().lower(),
    reason="This example has only been validated on MetaX C500",
)
def test_mla_decode_maca(heads, length, head_tile, splits):
    # Delay device-specific imports until the backend and device gates pass.
    from benchmark_mla_decode_maca import check_kernel
    from example_mla_decode_maca import mla_decode_maca

    compiled = mla_decode_maca(1, length, heads, num_split=splits, head_tile=head_tile)

    def run(q, q_pe, kv, k_pe):
        output = torch.empty_like(q)
        compiled(q, q_pe, kv, k_pe, output)
        return output

    cases = [("random", 7), ("random", 19), ("early-peak", 7), ("late-peak", 7), ("large-v", 7)]
    check_kernel("candidate", run, 1, length, cases, torch.device("cuda"), trace=False, heads=heads)
