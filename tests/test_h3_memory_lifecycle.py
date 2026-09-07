from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from musubi_tuner import minimax_h3_generate_video as gen
from musubi_tuner.modules import custom_offloading_utils as offload


def test_initial_image_encoder_does_not_retain_vae(monkeypatch):
    from musubi_tuner import minimax_h3_cache_latents as cache
    from musubi_tuner import minimax_h3_cache_mfi as mfi

    refs = []

    def load(*args, **kwargs):
        vae = nn.Linear(1, 1)
        refs.append(weakref.ref(vae))
        return vae

    monkeypatch.setattr(gen, "load_video_vae", load)
    monkeypatch.setattr(mfi, "read_frame", lambda *a: torch.zeros(2, 2, 3))
    monkeypatch.setattr(cache, "_prepare_pixels", lambda x: x)
    monkeypatch.setattr(cache, "_encode_condition_video", lambda *a: torch.ones(1, 24, 1, 1, 1))
    args = SimpleNamespace(h3_initial_image=["a", "b"], width=2, height=2, video_vae="unused", disable_numpy_memmap=False)
    source = gen._encode_initial_images(args, torch.device("cpu"))
    assert source.shape == (1, 24, 2, 1, 1)
    assert refs[0]() is None


def test_release_shared_vaes_preserves_other_models():
    shared = gen.H3SharedModels(device=torch.device("cpu"))
    shared.video_vaes[torch.float32] = nn.Linear(1, 1)
    shared.audio_vae = nn.Linear(1, 1)
    refs = [weakref.ref(shared.video_vaes[torch.float32]), weakref.ref(shared.audio_vae)]
    shared.transformer = object()
    shared.release_vaes()
    assert all(ref() is None for ref in refs)
    assert shared.transformer is not None


def test_prepare_evicts_tail_before_loading_head_and_skips_cpu_weights(monkeypatch):
    events = []

    class Block(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.linear = nn.Linear(2, 2)

        def to(self, device):
            events.append(("move", self.index))
            if self.index >= 2:
                assert self.linear.weight is None
            return self

    blocks = [Block(i) for i in range(4)]
    params = [b.linear.weight for b in blocks]
    worker = offload.ModelOffloader("test", blocks, 4, 2, False, torch.device("cpu"))
    worker.futures[3] = object()

    def wait(index):
        events.append(("wait", index))
        worker.futures.pop(index)

    monkeypatch.setattr(worker, "_wait_blocks_move", wait)
    monkeypatch.setattr(offload, "weighs_to_device", lambda b, d: events.append(("weights", b.index)))
    worker.prepare_block_devices_before_forward(blocks)
    assert events[:3] == [("wait", 3), ("weights", 2), ("weights", 3)]
    assert events.index(("move", 0)) > events.index(("weights", 3))
    assert all(b.linear.weight is p for b, p in zip(blocks, params))
    worker.thread_pool.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generic_prepare_twice_preserves_weights_and_resident_buffers():
    blocks = [nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8)) for _ in range(4)]
    snapshots = [{k: v.clone() for k, v in b.state_dict().items()} for b in blocks]
    worker = offload.ModelOffloader("test", blocks, 4, 2, False, torch.device("cuda"))
    try:
        for _ in range(2):
            worker.prepare_block_devices_before_forward(blocks)
            for i, block in enumerate(blocks):
                assert block[0].weight.device.type == ("cuda" if i < 2 else "cpu")
                assert block[0].bias.device.type == "cuda"
                assert block[1].weight.device.type == "cuda"
                for key, value in block.state_dict().items():
                    assert torch.equal(value.cpu(), snapshots[i][key])
            # Mimic a completed forward with trailing blocks resident.
            for block in blocks[:2]:
                offload.weighs_to_device(block, torch.device("cpu"))
            for block in blocks[2:]:
                offload.weighs_to_device(block, torch.device("cuda"))
    finally:
        worker.thread_pool.shutdown()
