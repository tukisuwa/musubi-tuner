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
        samples_per_video=20,
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
    assert {len(r["targets"]) for r in records} == {1, 2, 3, 4}
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


@pytest.mark.parametrize(
    "route,qwen,dit",
    [("dual", True, True), ("qwen_image_only", True, False), ("dit_latent_only", False, True), ("text_only", False, False)],
)
def test_training_and_generation_reference_routes(tmp_path, monkeypatch, route, qwen, dit):
    import musubi_tuner.minimax_h3_train_network as train
    import musubi_tuner.minimax_h3_generate_video as gen
    from musubi_tuner.minimax_h3.media import H3Record, H3Reference

    record = H3Record(
        video_path=tmp_path / "target.png",
        caption="test",
        references=(H3Reference(type="image", path=tmp_path / "ref.png"),),
        jsonl_line=1,
    )
    args = SimpleNamespace(
        task="ref2va",
        h3_reference_route=route,
        h3_independent_target_roles=True,
        h3_target_frame_indices="-3,7",
        h3_visual_condition_frame_indices="0",
        sample_prompts=str(tmp_path / "prompts.txt"),
        video_vae="mock",
        audio_vae="mock",
        text_encoder="mock",
    )
    monkeypatch.setattr(train, "_require_sampling_path", lambda *a: None)
    monkeypatch.setattr(train, "parse_inline_references", lambda *a: record.references)
    monkeypatch.setattr(train, "load_prompts", lambda *a: [dict(prompt="test", ref=["ref.png"], enum=0, width=32, height=32)])
    monkeypatch.setattr(train, "load_generation_record", lambda *a: record)
    monkeypatch.setattr(train, "decode_generation_visuals", lambda *a: ({}, {}))
    monkeypatch.setattr(train, "load_h3_processor", lambda *a: None)
    monkeypatch.setattr(train, "load_h3_text_encoder", lambda *a, **k: torch.nn.Identity())
    presentations = []

    def presentation(rec, task, visuals):
        presentations.append((task, bool(rec.references)))
        return object()

    monkeypatch.setattr(train, "build_presentation", presentation)
    monkeypatch.setattr(train, "encode_h3_presentation", lambda *a: (torch.zeros(2, 4), torch.tensor([1, 0 if qwen else 1])))

    class VAE(torch.nn.Module):
        vae_ratio = 16

    monkeypatch.setattr(train, "load_video_vae", lambda *a, **k: VAE())
    monkeypatch.setattr(train, "load_audio_vae", lambda *a, **k: VAE())
    geometry = H3VideoGeometry(1, 2, 2)
    conditions = (torch.zeros(1, 24, 1, 2, 2),)
    monkeypatch.setattr(train, "encode_visual_conditions", lambda *a: (conditions, (), {0: geometry}))
    samples, _ = train.MiniMaxH3NetworkTrainer().prepare_sampling(args, SimpleNamespace(device=torch.device("cpu")), None)
    assert presentations == [("ref2va" if qwen else "t2va", qwen)]
    assert bool(samples[0]["h3_visual_conditions"]) == dit
    assert samples[0]["h3_layout"].task == ("ref2va" if dit else "t2va")

    gen_args = gen.setup_parser().parse_args(
        [
            "--output",
            str(tmp_path),
            "--task",
            "ref2va",
            "--h3_reference_route",
            route,
            "--h3_independent_target_roles",
            "--h3_target_frame_indices=-3,7",
            "--h3_visual_condition_frame_indices=0",
            "--width",
            "32",
            "--height",
            "32",
        ]
    )
    gen_args.text_cache = None
    monkeypatch.setattr(gen, "build_presentation", presentation)
    monkeypatch.setattr(gen, "load_h3_processor", lambda *a: None)
    monkeypatch.setattr(gen, "load_h3_text_encoder", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(gen, "encode_h3_presentation", lambda *a: (torch.zeros(2, 4), torch.tensor([1, 0 if qwen else 1])))
    gen._encode_text(gen_args, record, {}, torch.device("cpu"))
    assert presentations[-1] == ("ref2va" if qwen else "t2va", qwen)

    @contextmanager
    def borrowed(*a):
        yield VAE()

    monkeypatch.setattr(gen, "_borrowed_video_vae", borrowed)
    monkeypatch.setattr(gen, "encode_visual_conditions", lambda *a: (conditions, (), {0: geometry}))
    visual, geoms, refs, audio = gen._encode_conditions(gen_args, record, {}, None, torch.device("cpu"))
    assert bool(visual) == dit and not audio
    layout = gen._build_layout(gen_args, 2, geoms, refs)
    assert layout.task == ("ref2va" if dit else "t2va")


@pytest.mark.parametrize("role", ["first", "last"])
def test_mfi_training_sample_accepts_single_fl_control(role, monkeypatch):
    import musubi_tuner.minimax_h3_train_network as train

    monkeypatch.setattr(train, "_require_sampling_path", lambda *a: None)
    args = SimpleNamespace(
        task="fl2va", h3_independent_target_roles=True, h3_target_frame_indices="-3,7", h3_visual_condition_frame_indices="0"
    )
    sample = train._normalize_h3_sample_parameter(args, dict(prompt="test", **{f"{role}_frame": "image.png"}))
    assert sample[f"{role}_frame"] == "image.png"


def test_legacy_batch_mfi_decode_needs_no_audio_vae(tmp_path, monkeypatch):
    import musubi_tuner.minimax_h3_generate_video as gen

    args = gen.setup_parser().parse_args(
        ["--output", str(tmp_path), "--latent_path", "mock.safetensors", "--output_type", "images"]
    )
    monkeypatch.setattr(
        gen,
        "_load_latent_file",
        lambda *a: (
            torch.zeros(1, 24, 2, 2, 2),
            torch.zeros(1, 32, 2, 2),
            124,
            {"h3_independent_target_roles": "true", "h3_target_frame_indices": "-3,7"},
        ),
    )
    outputs = []
    monkeypatch.setattr(gen, "_decode_and_save", lambda args, video, audio, *rest: outputs.append(audio))
    gen.process_latent_decode(args, torch.device("cpu"))
    assert outputs == [None]


def test_text_only_prompt_without_references(tmp_path):
    import musubi_tuner.minimax_h3_generate_video as gen
    from musubi_tuner.minimax_h3.generation_inputs import load_generation_record

    args = gen.setup_parser().parse_args(
        [
            "--output",
            str(tmp_path),
            "--task",
            "ref2va",
            "--prompt",
            "test",
            "--h3_reference_route",
            "text_only",
            "--h3_independent_target_roles",
            "--h3_target_frame_indices=-3,7",
            "--output_type",
            "latent",
        ]
    )
    gen.validate_prompt_args(args)
    assert load_generation_record(args).references == ()
    args.task = "t2va"
    with pytest.raises(ValueError, match="Ref2VA base"):
        gen.validate_prompt_args(args)
