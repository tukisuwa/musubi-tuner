import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from safetensors import safe_open

from musubi_tuner import minimax_h3_generate_video as gen
from musubi_tuner import minimax_h3_train_network as train
from musubi_tuner import minimax_h3_cache_mfi as cache_mfi
from musubi_tuner.minimax_h3 import generation_inputs as inputs
from musubi_tuner.minimax_h3.packing import FRAME_RESCALE, H3VideoGeometry, build_position_grid
from musubi_tuner.minimax_h3.text_encoder import build_presentation


def _images(tmp_path, count=3):
    paths = []
    for index in range(count):
        path = tmp_path / f"control{index}.png"
        Image.new("RGB", (32, 32), (index * 20, 0, 0)).save(path)
        paths.append(str(path))
    return paths


def _generation_args(tmp_path, paths, targets):
    options = [
        "--task",
        "fl2va",
        "--prompt",
        "test",
        "--output",
        str(tmp_path / "out"),
        "--output_type",
        "latent",
        "--width",
        "32",
        "--height",
        "32",
    ]
    for path in paths:
        options += ["--condition_image", path]
    if targets is None:
        options += [
            "--frame_count",
            "1",
            "--one_frame",
            "target_index=24,control_index=" + ";".join(str(48 * i) for i in range(len(paths))),
        ]
    else:
        options += [
            "--h3_independent_target_roles",
            "--h3_target_frame_indices=" + targets,
            "--h3_visual_condition_frame_indices=" + ",".join(str(48 * i) for i in range(len(paths))),
        ]
    return gen.setup_parser().parse_args(options)


@pytest.mark.parametrize("count", [1, 2, 3, 5])
@pytest.mark.parametrize("targets", [None, "24", "-24,72"])
def test_generation_ordered_conditions_and_coordinates(tmp_path, count, targets):
    paths = _images(tmp_path, count)
    args = _generation_args(tmp_path, paths, targets)
    gen.validate_prompt_args(args)
    record = inputs.load_generation_record(args)
    raw, text = inputs.decode_generation_visuals(args, record, None)
    assert list(raw) == [f"cond_{i:03d}" for i in range(count)]
    presentation = build_presentation(record, "fl2va", text)
    assert len(presentation.images) == count
    for i, pixels in enumerate(presentation.images):
        assert int(pixels[0, 0, 0]) == i * 20
        assert f"<Picture {i + 1}>" in presentation.text
    layout = gen._build_layout(args, 2, (H3VideoGeometry(1, 2, 2),) * count, ())
    grid = build_position_grid(layout)
    for i in range(count):
        assert torch.all(grid[layout.segment(f"cond_{i:03d}").row_slice, 0] == 2 + FRAME_RESCALE * 48 * i)
    assert layout.target_video.frames == (1 if targets is None else len(targets.split(",")))


def test_generation_text_cache_identity_keeps_condition_order(tmp_path):
    args = _generation_args(tmp_path, _images(tmp_path), "24")
    record = inputs.load_generation_record(args)

    def identity():
        _, visuals = inputs.decode_generation_visuals(args, record, None)
        return gen._text_conditioning_cache_key(args, record, build_presentation(record, "fl2va", visuals))

    original = identity()
    args.condition_image.reverse()
    assert identity() != original
    args.condition_image.reverse()
    assert identity() == original


def test_generation_rejects_mixed_and_mismatched_conditions(tmp_path):
    args = _generation_args(tmp_path, _images(tmp_path), "24,72")
    args.h3_visual_condition_frame_indices = "0,48"
    with pytest.raises(ValueError, match="entry per condition"):
        gen.validate_prompt_args(args)
    args.h3_visual_condition_frame_indices = "0,48,96"
    args.first_frame = args.condition_image[0]
    with pytest.raises(ValueError, match="not both"):
        gen.validate_prompt_args(args)
    args.first_frame = None
    args.task = "ref2va"
    args.ref = [args.condition_image[0]]
    with pytest.raises(ValueError, match="does not accept"):
        gen.validate_prompt_args(args)


@pytest.mark.parametrize("targets", [None, "24", "-24,72"])
def test_training_sample_uses_real_ordered_frontend(tmp_path, monkeypatch, targets):
    paths = _images(tmp_path)
    args = SimpleNamespace(
        task="fl2va",
        h3_independent_target_roles=targets is not None,
        h3_target_frame_indices=targets,
        h3_visual_condition_frame_indices="0,48,96",
        sample_prompts=str(tmp_path / "prompts.txt"),
        video_vae="mock",
        audio_vae="mock",
        text_encoder="mock",
    )
    sample = dict(prompt="test", control_image_path=paths, enum=0, width=32, height=32, frame_count=1)
    if targets is None:
        sample["one_frame"] = "target_index=24,control_index=0;48;96"
    monkeypatch.setattr(train, "_require_sampling_path", lambda *a: None)
    monkeypatch.setattr(train, "load_prompts", lambda *a: [sample])
    monkeypatch.setattr(train, "load_h3_processor", lambda *a: None)
    monkeypatch.setattr(train, "load_h3_text_encoder", lambda *a, **k: torch.nn.Identity())
    presentations = []

    def encode_text(processor, model, presentation):
        presentations.append(presentation)
        return torch.zeros(2, 12), torch.tensor([0, 1])

    monkeypatch.setattr(train, "encode_h3_presentation", encode_text)

    class VAE(torch.nn.Module):
        vae_ratio = 16

    monkeypatch.setattr(train, "load_video_vae", lambda *a, **k: VAE())
    monkeypatch.setattr(train, "load_audio_vae", lambda *a, **k: VAE())
    monkeypatch.setattr(inputs, "encode_video_condition", lambda vae, pixels: torch.full((1, 24, 1, 2, 2), float(pixels.mean())))
    samples, _ = train.MiniMaxH3NetworkTrainer().prepare_sampling(args, SimpleNamespace(device=torch.device("cpu")), None)
    result = samples[0]
    assert len(presentations[0].images) == 3
    assert len(result["h3_visual_conditions"]) == 3
    assert [float(t.mean()) for t in result["h3_visual_conditions"]] == sorted(
        float(t.mean()) for t in result["h3_visual_conditions"]
    )
    roles = [s.role for s in result["h3_layout"].segments if s.kind == "visual_condition"]
    assert roles == ["cond_000", "cond_001", "cond_002"]
    if targets is not None:
        assert result["h3_layout"].target_frame_indices == tuple(map(int, targets.split(",")))
        assert result["h3_layout"].visual_condition_frame_indices == (0, 48, 96)


@pytest.mark.parametrize("target_indices", [[24], [-24, 72]])
def test_mfi_fl2va_cache_cli_stages_to_runtime(tmp_path, monkeypatch, target_indices):
    import sys
    from musubi_tuner import minimax_h3_cache_latents as latents
    from musubi_tuner.minimax_h3 import video_vae, audio_vae, text_encoder
    from musubi_tuner.dataset.bucket import BucketBatchManager
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

    paths = _images(tmp_path)
    record = dict(
        id="sample",
        caption="test",
        targets=[dict(path=paths[0], index=i) for i in target_indices],
        controls=[dict(path=path, index=48 * i) for i, path in enumerate(paths)],
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    output = tmp_path / "cache"
    monkeypatch.setattr(latents, "fingerprint_checkpoint", lambda *a: "test")
    monkeypatch.setattr(audio_vae, "load_audio_vae", lambda *a, **k: object())
    monkeypatch.setattr(video_vae, "load_video_vae", lambda *a, **k: object())
    monkeypatch.setattr(latents, "encode_one_frame_silence_latent", lambda *a: torch.zeros(32, 2, 2))
    monkeypatch.setattr(latents, "_encode_target_video", lambda *a: torch.ones(1, 24, 1, 2, 2))
    monkeypatch.setattr(latents, "_encode_condition_video", lambda vae, pixels: torch.full((1, 24, 1, 2, 2), float(pixels.mean())))
    monkeypatch.setattr(text_encoder, "load_h3_processor", lambda *a: None)
    monkeypatch.setattr(text_encoder, "load_h3_text_encoder", lambda *a, **k: None)
    seen = []

    def encode_text(processor, model, presentation):
        seen.append(presentation)
        return torch.zeros(2, 5120), torch.tensor([0, 1])

    monkeypatch.setattr(text_encoder, "encode_h3_presentation", encode_text)
    base = [
        "cache_mfi",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--task",
        "fl2va",
        "--width",
        "32",
        "--height",
        "32",
        "--video_vae",
        paths[0],
        "--audio_vae",
        paths[0],
        "--text_encoder",
        paths[0],
        "--device",
        "cpu",
    ]
    for stage in ("latents", "text"):
        monkeypatch.setattr(sys, "argv", base + ["--stage", stage])
        cache_mfi.main()
    assert len(seen[0].images) == 3
    assert "<Picture 3>" in seen[0].text
    key, latent_path, text_path = cache_mfi.cache_paths(output, record, 32, 32)
    with safe_open(str(latent_path), framework="pt") as handle:
        assert "latents_cond_002_1x2x2_float32" in handle.keys()
        assert json.loads(handle.metadata()["mfi_contract"])["task"] == "fl2va"
    item = ItemInfo(key, "test", (32, 32), (32, 32))
    item.latent_cache_path = str(latent_path)
    item.text_encoder_output_cache_path = str(text_path)
    batch = BucketBatchManager({(32, 32): [item]}, 1)[0]
    runtime = train._runtime_batch_plan(batch, batch["latents"], independent_target_roles=True)
    assert runtime.layout.task == "fl2va"
    assert runtime.layout.target_frame_indices == tuple(target_indices)
    assert runtime.layout.visual_condition_frame_indices == (0, 48, 96)
    assert len(runtime.visual_conditions) == 3
    train._validate_reference_route(runtime, SimpleNamespace(task="fl2va", h3_reference_route="native"))
    # Both stages must use the same family; a Ref2VA text stage cannot pair with FL2VA latents.
    other_output = tmp_path / "other"
    other_output.mkdir()
    _, other_latent, _ = cache_mfi.cache_paths(other_output, record, 32, 32)
    other_latent.write_bytes(latent_path.read_bytes())
    wrong = list(base)
    wrong[wrong.index("fl2va")] = "ref2va"
    wrong[wrong.index(str(output))] = str(other_output)
    monkeypatch.setattr(sys, "argv", wrong + ["--stage", "text"])
    with pytest.raises(ValueError, match="contracts differ"):
        cache_mfi.main()


def test_fl2va_cache_has_no_ref2va_nine_control_cap(tmp_path):
    paths = _images(tmp_path, 10)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(dict(targets=[dict(path=paths[0], index=24)], controls=[dict(path=p, index=i) for i, p in enumerate(paths)]))
        + "\n"
    )
    args = cache_mfi.setup_parser().parse_args(
        ["--manifest", str(manifest), "--output", str(tmp_path / "cache"), "--task", "fl2va", "--stage", "latents"]
    )
    assert len(cache_mfi.read_records(args)[0]["controls"]) == 10
    args.task = "ref2va"
    with pytest.raises(ValueError, match="at most 9"):
        cache_mfi.read_records(args)
    args.task = "fl2va"
    args.route = "dual"
    with pytest.raises(ValueError, match="does not support"):
        cache_mfi.read_records(args)
