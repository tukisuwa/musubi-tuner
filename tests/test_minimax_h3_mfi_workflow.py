from contextlib import contextmanager
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.minimax_h3.mfi import plan_indices, cached_indices, mask_condition
from musubi_tuner.minimax_h3.packing import build_h3_layout, H3VideoGeometry
from musubi_tuner.minimax_h3.sampling import sample_joint_av


def test_interpolation_keeps_control_overlap_and_requested_order():
    plan = plan_indices((9, 3, -3), (3, 6), relative=True, interpolate=True, thresholds=(2, 2))
    assert plan.controls == (0, 3)
    assert 3 not in plan.targets  # pure control@6 is not generated
    assert 0 in plan.targets  # target@3 and control@3 coexist
    assert plan.output_indices == (9, 3, -3)
    assert tuple(plan.targets[i] for i in plan.output_slots) == (6, 0, -6)
    dense = plan_indices((9, -3), (3, 6), interpolate=True, thresholds=(2, 2), save_interpolated=True)
    assert dense.output_indices == tuple(sorted(dense.output_indices))
    assert 3 not in dense.targets and 6 not in dense.targets


@pytest.mark.parametrize(
    "targets,controls,kwargs",
    [
        ((), (0,), {}),
        ((0, 0), (1,), {}),
        ((0,), (), {"relative": True}),
        ((0,), (1,), {"thresholds": (0, 1)}),
        ((False,), (), {}),
    ],
)
def test_invalid_plans_fail(targets, controls, kwargs):
    with pytest.raises(ValueError):
        plan_indices(targets, controls, **kwargs)


def test_cache_indices_are_authoritative_and_conflicts_fail():
    batch = {"mfi_target_indices": torch.tensor([[-10, 8]])}
    assert cached_indices(batch, "mfi_target_indices", None) == (-10, 8)
    with pytest.raises(ValueError, match="conflict"):
        cached_indices(batch, "mfi_target_indices", (0, 1))


def test_strength_initialization_and_sigma_match():
    layout = build_h3_layout(
        task="t2va",
        text_length=1,
        target_video=H3VideoGeometry(3, 2, 2),
        target_audio_frames=2,
        independent_target_roles=True,
        target_frame_indices=(-4, 0, 9),
    )
    noise = torch.ones(1, 24, 3, 2, 2)
    source = torch.full_like(noise, 4)
    calls = []

    def model(**kw):
        calls.append(kw)
        return SimpleNamespace(video=torch.zeros_like(kw["video_latents"]), audio=torch.zeros_like(kw["audio_latents"]))

    result = sample_joint_av(
        model,
        layout=layout,
        text_hidden_states=torch.zeros(1, 1, 2),
        text_token_tags=torch.ones(1, 1, dtype=torch.int64),
        initial_video=noise,
        initial_audio=torch.ones(1, 32, 2, 2),
        initial_video_source=source,
        strength=0.5,
        steps=2,
    )
    assert float(calls[0]["model_t_video"]) == pytest.approx(1 / 13)
    assert float(calls[0]["model_t_audio"]) == pytest.approx(0.25)
    torch.testing.assert_close(result.video, source / 13 + noise * (12 / 13))


def test_mask_resamples_to_latent_grid():
    latent = torch.ones(1, 24, 1, 2, 2)
    mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    masked = mask_condition(latent, mask)
    assert masked[..., 0].count_nonzero() == 0
    assert torch.equal(masked[..., 1], latent[..., 1])


def test_grouped_image_and_video_selection(tmp_path, monkeypatch):
    import musubi_tuner.dataset.media_utils as media
    from musubi_tuner.minimax_h3_cache_mfi import read_records

    for index in (-3, 0, 7):
        Image.new("RGB", (32, 32)).save(tmp_path / f"example_{index}.png")
    args = SimpleNamespace(
        manifest=None,
        image_directory=str(tmp_path),
        video_directory=None,
        control_indices=None,
        target_indices=None,
        caption="test",
        route="dual",
        relative=False,
        seed=42,
        samples_per_video=2,
        num_controls=2,
        max_frame_distance=3,
        max_targets=4,
    )
    records = read_records(args)
    assert [e["index"] for e in records[0]["targets"]] == [-3, 7]
    assert records[0]["controls"][0]["index"] == 0
    (tmp_path / "clip.mp4").touch()
    monkeypatch.setattr(media, "load_video", lambda *a, **k: list(range(20)))
    args.image_directory = None
    args.video_directory = str(tmp_path)
    records = read_records(args)
    assert records == read_records(args)
    for record in records:
        controls = [e["index"] for e in record["controls"]]
        assert len(controls) == 2
        for target in record["targets"]:
            assert target["index"] == target["frame"]
            assert target["index"] not in controls
            assert min(abs(target["index"] - c) for c in controls) <= 3


def test_prompt_accepts_mfi_image_masks_and_thresholds():
    from musubi_tuner.minimax_h3_generate_video import parse_prompt_line

    parsed = parse_prompt_line(
        "test --h3_initial_image first.png --h3_control_mask a.png --h3_control_mask b.png --h3_interpolation_thresholds 2 3"
    )
    assert parsed["h3_initial_image"] == ["first.png"]
    assert parsed["h3_control_mask"] == ["a.png", "b.png"]
    assert parsed["h3_interpolation_thresholds"] == (2, 3)


def test_training_sample_uses_explicit_mfi_coordinates():
    from musubi_tuner.minimax_h3_train_network import _normalize_h3_sample_parameter

    args = SimpleNamespace(
        task="t2va",
        h3_independent_target_roles=True,
        h3_target_frame_indices=None,
        h3_visual_condition_frame_indices=None,
        sample_prompts=None,
    )
    sample = _normalize_h3_sample_parameter(args, {"prompt": "test", "h3_target_frame_indices": "-3,7"})
    assert sample["frame_count"] == 2
    assert sample["h3_target_frame_indices"] == (-3, 7)
    with pytest.raises(ValueError, match="explicit target indices"):
        _normalize_h3_sample_parameter(args, {"prompt": "test"})


def test_cache_builder_to_training_runtime(tmp_path, monkeypatch):
    import musubi_tuner.minimax_h3_cache_latents as cache
    from musubi_tuner.minimax_h3_cache_mfi import build_mfi_tensors
    from musubi_tuner.dataset.cache_io import save_latent_cache_minimax_h3
    from musubi_tuner.dataset.image_video_dataset import ItemInfo
    from musubi_tuner.minimax_h3_train_network import _runtime_batch_plan
    from musubi_tuner.utils.model_utils import remove_dtype_suffix

    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "red").save(image)
    mask = tmp_path / "mask.png"
    Image.new("L", (32, 32), 0).save(mask)
    monkeypatch.setattr(cache, "_encode_target_video", lambda *args: torch.ones(1, 24, 1, 2, 2))
    monkeypatch.setattr(cache, "_encode_condition_video", lambda *args: torch.ones(1, 24, 1, 2, 2))
    record = dict(
        id="a",
        targets=[dict(path=str(image), index=-3), dict(path=str(image), index=7)],
        controls=[dict(path=str(image), index=2, mask=str(mask))],
    )
    tensors = build_mfi_tensors(record, video_vae=None, silence=torch.zeros(32, 2, 2), size=(32, 32), seed=1, relative=True)
    item = ItemInfo("a", "", (32, 32), (32, 32))
    item.latent_cache_path = str(tmp_path / "cache.safetensors")
    save_latent_cache_minimax_h3(item, tensors)
    batch = {}
    for key, value in load_file(item.latent_cache_path).items():
        name = remove_dtype_suffix(key)
        if name.startswith("latents_"):
            name = name.rsplit("_", 1)[0]
        batch[name] = value.unsqueeze(0)
    batch.update(mmh3_hidden_states=[torch.zeros(2, 5120)], mmh3_token_tags=[torch.tensor([0, 1])])
    runtime = _runtime_batch_plan(batch, batch["latents"], independent_target_roles=True)
    assert runtime.layout.target_frame_indices == (-5, 5)
    assert runtime.layout.visual_condition_frame_indices == (0,)
    assert runtime.visual_conditions[0].count_nonzero() == 0
    assert runtime.audio_present.item() == 0


def test_saved_mfi_latent_roundtrip_and_selected_decode(tmp_path, monkeypatch):
    import musubi_tuner.minimax_h3_generate_video as gen

    args = gen.setup_parser().parse_args(["--output", str(tmp_path), "--task", "ref2va"])
    args.h3_independent_target_roles = True
    args.h3_target_frame_indices = "8,-2"
    args.h3_visual_condition_frame_indices = "0"
    args._h3_mfi_plan = plan_indices((8, -2), (0,), interpolate=True, save_interpolated=False)
    n = len(args._h3_mfi_plan.targets)
    video = torch.stack([torch.full((1, 24, 2, 2), float(i)) for i in range(n)], dim=2)
    path = tmp_path / "latent.safetensors"
    gen._save_latent_file(path, video, None, args, 1)
    decoded, audio, _, meta = gen._load_latent_file(path)
    assert audio is None and torch.equal(decoded, video)
    assert "h3_mfi_plan" in meta
    slots = []

    class VAE:
        def decode(self, value):
            slots.append(int(value[0, 0, 0, 0, 0]))
            return torch.zeros(1, 3, 1, 32, 32)

    @contextmanager
    def borrowed(*args):
        yield VAE()

    monkeypatch.setattr(gen, "_borrowed_video_vae", borrowed)
    sequences = []
    args.h3_sequence_video = True
    monkeypatch.setattr(gen, "write_video_only", lambda frames, path, fps: sequences.append((frames, fps)))
    gen._decode_and_save(args, video, None, tmp_path / "out", torch.device("cpu"))
    assert tuple(slots) == args._h3_mfi_plan.output_slots
    assert (tmp_path / "out/000_index_+8.png").is_file()
    assert (tmp_path / "out/001_index_-2.png").is_file()
    assert sequences[0][0].shape == (11, 32, 32, 3)
    assert sequences[0][1] == 24
