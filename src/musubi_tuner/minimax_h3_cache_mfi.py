"""Build H3 MFI image-slice caches from indexed image groups or sampled videos.

Each JSONL record carries targets/controls as lists of {path,index,frame?,mask?}.
`frame` addresses a video decoded at 24 fps; `index` is the independent model coordinate.
Run latent and text stages in separate processes to release their model families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch

from musubi_tuner.minimax_h3.mfi import plan_indices, mask_condition


def read_records(args):
    base = Path(args.manifest).resolve().parent if args.manifest else Path.cwd()
    if args.manifest:
        records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    elif args.image_directory:
        groups = {}
        for path in sorted(Path(args.image_directory).iterdir()):
            match = re.fullmatch(r"(.+)_(-?\d+)\.(png|jpg|jpeg|webp)", path.name, re.I)
            if match:
                groups.setdefault(match[1], {})[int(match[2])] = str(path.resolve())
        records = []
        for name, frames in groups.items():
            controls = args.control_indices or [sorted(frames)[len(frames) // 2]]
            targets = args.target_indices or [i for i in sorted(frames) if i not in controls]
            caption = Path(args.image_directory) / f"{name}.txt"
            records.append(
                dict(
                    id=name,
                    caption=caption.read_text() if caption.exists() else args.caption,
                    targets=[dict(path=frames[i], index=i) for i in targets],
                    controls=[dict(path=frames[i], index=i) for i in controls],
                )
            )
    else:
        from musubi_tuner.dataset.media_utils import load_video

        records = []
        rng = random.Random(args.seed)
        for path in sorted(Path(args.video_directory).iterdir()):
            if path.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm"}:
                continue
            count = len(load_video(str(path), target_fps=24, fps_resample_mode="timestamps"))
            for draw in range(args.samples_per_video):
                controls = args.control_indices or sorted(rng.sample(range(count), min(args.num_controls, count)))
                candidates = [
                    i for i in range(count) if i not in controls and any(abs(i - c) <= args.max_frame_distance for c in controls)
                ]
                targets = args.target_indices or sorted(rng.sample(candidates, min(args.max_targets, len(candidates))))
                caption = path.with_suffix(".txt")
                records.append(
                    dict(
                        id=f"{path.stem}_{draw}",
                        caption=caption.read_text() if caption.exists() else args.caption,
                        targets=[dict(path=str(path.resolve()), frame=i, index=i) for i in targets],
                        controls=[dict(path=str(path.resolve()), frame=i, index=i) for i in controls],
                    )
                )
    seen = set()
    for number, record in enumerate(records):
        record.setdefault("id", f"item{number:06d}")
        if not re.fullmatch(r"[\w.-]+", record["id"]) or record["id"] in seen:
            raise ValueError("MFI record IDs must be unique safe filenames")
        seen.add(record["id"])
        record.setdefault("controls", [])
        record.setdefault("caption", args.caption)
        if args.route in {"dual", "dit_latent_only", "qwen_image_only"} and not record["controls"]:
            raise ValueError(f"MFI {args.route} requires controls")
        if len(record["controls"]) > 9:
            raise ValueError("H3 Ref2VA supports at most 9 image controls")
        for entry in record["targets"] + record["controls"]:
            for key in ("path", "mask"):
                if key in entry:
                    path = (base / entry[key]).resolve()
                    if not path.is_file():
                        raise ValueError(f"Missing MFI {key}: {path}")
                    entry[key] = str(path)
            if "frame" in entry and (type(entry["frame"]) is not int or entry["frame"] < 0):
                raise ValueError("MFI source frame must be a nonnegative integer")
        plan_indices(
            [e["index"] for e in record["targets"]],
            [e["index"] for e in record["controls"]],
            relative=record.get("relative", args.relative),
        )
    if not records:
        raise ValueError("No MFI records found")
    return records


def cache_paths(output, record, width, height):
    key = f"{record['id']}_00000-{len(record['targets']):03d}"
    return key, output / f"{key}_{width:04d}x{height:04d}_mmh3.safetensors", output / f"{key}_mmh3_te.safetensors"


def read_frame(entry, size, videos):
    if "frame" in entry:
        from musubi_tuner.dataset.media_utils import load_video

        if entry["path"] not in videos:
            videos[entry["path"]] = load_video(entry["path"], target_fps=24, fps_resample_mode="timestamps")
        frames = videos[entry["path"]]
        if entry["frame"] >= len(frames):
            raise ValueError("MFI source frame exceeds decoded video length")
        image = Image.fromarray(frames[entry["frame"]]).convert("RGB")
    else:
        image = Image.open(entry["path"]).convert("RGB")
    return torch.from_numpy(np.array(image.resize(size, Image.Resampling.LANCZOS)))


def build_mfi_tensors(record, *, video_vae, silence, size, seed, relative=False, route="dual"):
    from musubi_tuner.minimax_h3_cache_latents import (
        _prepare_pixels,
        _encode_target_video,
        _encode_condition_video,
        _visual_key,
        _audio_key,
    )

    plan = plan_indices(
        [e["index"] for e in record["targets"]], [e["index"] for e in record["controls"]], relative=record.get("relative", relative)
    )
    videos = {}
    targets = [
        _encode_target_video(video_vae, _prepare_pixels(read_frame(entry, size, videos)[None]), seed, f"{record['id']}#{slot}")[
            0
        ].cpu()
        for slot, entry in enumerate(record["targets"])
    ]
    if any(t.shape != (24, 1, size[1] // 16, size[0] // 16) for t in targets):
        raise ValueError("MFI requires each target to encode to one image latent slice")
    target = torch.cat(targets, dim=1)
    tensors = {
        _visual_key("", target): target,
        _audio_key("", silence): silence,
        "audio_present_float32": torch.tensor(0.0),
        "mfi_target_indices_int64": torch.tensor(plan.targets, dtype=torch.int64),
    }
    controls = record["controls"] if route in {"dual", "dit_latent_only"} else []
    if route in {"dual", "dit_latent_only"} and not controls:
        raise ValueError(f"MFI {route} requires controls")
    tensors["mfi_control_indices_int64"] = torch.tensor(plan.controls if controls else (), dtype=torch.int64)
    for slot, entry in enumerate(controls):
        latent = _encode_condition_video(video_vae, _prepare_pixels(read_frame(entry, size, videos)[None]))
        if "mask" in entry:
            latent = mask_condition(latent, np.asarray(Image.open(entry["mask"]).convert("L"), dtype=np.float32) / 255)
        tensors[_visual_key(f"ref_{slot:03d}_image", latent[0])] = latent[0].cpu()
    if any(not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("MFI cache contains nonfinite tensors")
    return tensors


def main():
    from musubi_tuner.dataset.image_video_dataset import ItemInfo
    from musubi_tuner.dataset.cache_io import save_latent_cache_minimax_h3, save_text_encoder_output_cache_minimax_h3

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--image_directory")
    source.add_argument("--video_directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("latents", "text"), required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--caption", default="")
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--target_indices", nargs="+", type=int)
    parser.add_argument("--control_indices", nargs="+", type=int)
    parser.add_argument("--num_controls", type=int, default=1)
    parser.add_argument("--max_targets", type=int, default=4)
    parser.add_argument("--max_frame_distance", type=int, default=32)
    parser.add_argument("--samples_per_video", type=int, default=1)
    parser.add_argument("--route", choices=("dual", "dit_latent_only", "qwen_image_only", "text_only"), default="dual")
    parser.add_argument("--video_vae")
    parser.add_argument("--audio_vae")
    parser.add_argument("--text_encoder")
    parser.add_argument("--text_encoder_blocks_to_swap", type=int, default=0)
    parser.add_argument("--text_encoder_attn_mode", default="sdpa")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.width, args.height) <= 0 or args.width % 32 or args.height % 32:
        parser.error("size must be positive multiples of 32")
    if min(args.num_controls, args.max_targets, args.max_frame_distance, args.samples_per_video) <= 0:
        parser.error("video selection parameters must be positive")
    records = read_records(args)
    output = Path(args.output)
    required = ("video_vae", "audio_vae") if args.stage == "latents" else ("text_encoder",)
    for name in required:
        value = getattr(args, name)
        if not value or not Path(value).exists():
            parser.error(f"--{name} must be an existing checkpoint")
    # Validate all destinations and paired stage contracts before loading models.
    from safetensors import safe_open
    from musubi_tuner.minimax_h3_cache_latents import fingerprint_file, fingerprint_checkpoint

    contracts = {}
    for record in records:
        _, latent_path, text_path = cache_paths(output, record, args.width, args.height)
        destination, peer = (latent_path, text_path) if args.stage == "latents" else (text_path, latent_path)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite MFI cache: {destination}")
        media = {
            entry[key]: fingerprint_file(entry[key])
            for entry in record["targets"] + record["controls"]
            for key in ("path", "mask")
            if key in entry
        }
        contract = json.dumps(
            dict(
                record=record, route=args.route, relative=args.relative, seed=args.seed, size=[args.width, args.height], media=media
            ),
            sort_keys=True,
        )
        identity = hashlib.sha256(contract.encode()).hexdigest()
        if peer.exists():
            with safe_open(str(peer), framework="pt") as handle:
                if (handle.metadata() or {}).get("mfi_fingerprint") != identity:
                    raise ValueError(f"MFI latent/text stage contracts differ: {peer}")
        contracts[record["id"]] = {"mfi_contract": contract, "mfi_fingerprint": identity}
    model_metadata = {f"{name}_fingerprint": fingerprint_checkpoint(getattr(args, name)) for name in required}
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "latents":
        from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
        from musubi_tuner.minimax_h3.video_vae import load_video_vae, VIDEO_VAE_ENCODE_DTYPE
        from musubi_tuner.minimax_h3_cache_latents import encode_one_frame_silence_latent

        audio_vae = load_audio_vae(args.audio_vae, device=args.device, dtype=torch.float32)
        silence = encode_one_frame_silence_latent(audio_vae).cpu()
        del audio_vae
        from musubi_tuner.utils.device_utils import clean_memory_on_device

        clean_memory_on_device(torch.device(args.device))
        video_vae = load_video_vae(args.video_vae, device=args.device, dtype=VIDEO_VAE_ENCODE_DTYPE)
    else:
        from musubi_tuner.minimax_h3.text_encoder import (
            load_h3_processor,
            load_h3_text_encoder,
            encode_h3_presentation,
            build_presentation,
            H3TextVisual,
        )
        from musubi_tuner.minimax_h3.media import H3Record, H3Reference

        processor = load_h3_processor()
        encoder = load_h3_text_encoder(
            args.text_encoder,
            device=args.device,
            dtype=torch.bfloat16,
            blocks_to_swap=args.text_encoder_blocks_to_swap,
            attn_mode=args.text_encoder_attn_mode,
        )
    for record in records:
        key, latent_path, text_path = cache_paths(output, record, args.width, args.height)
        item = ItemInfo(key, record["caption"], (args.width, args.height), (args.width, args.height))
        item.latent_cache_path = str(latent_path)
        item.text_encoder_output_cache_path = str(text_path)
        destination = item.latent_cache_path if args.stage == "latents" else item.text_encoder_output_cache_path
        if Path(destination).exists():
            raise FileExistsError(f"Refusing to overwrite MFI cache: {destination}")
        metadata = contracts[record["id"]] | model_metadata
        if args.stage == "latents":
            tensors = build_mfi_tensors(
                record,
                video_vae=video_vae,
                silence=silence,
                size=(args.width, args.height),
                seed=args.seed,
                relative=args.relative,
                route=args.route,
            )
            save_latent_cache_minimax_h3(item, tensors, metadata)
        else:
            controls = record["controls"] if args.route in {"dual", "qwen_image_only"} else []
            if args.route == "qwen_image_only" and not controls:
                raise ValueError("Qwen-image-only route requires controls")
            # Synthetic unique paths identify separately sampled frames of the same video.
            refs = tuple(H3Reference(type="image", path=Path(f"mfi-control-{i}.png")) for i in range(len(controls)))
            videos = {}
            visuals = {
                ref.path: H3TextVisual(read_frame(entry, (args.width, args.height), videos)[None])
                for ref, entry in zip(refs, controls)
            }
            h3_record = H3Record(
                video_path=Path(record["targets"][0]["path"]), caption=record["caption"], references=refs, jsonl_line=1
            )
            presentation = build_presentation(h3_record, "ref2va" if refs else "t2va", visuals)
            hidden, tags = encode_h3_presentation(processor, encoder, presentation)
            save_text_encoder_output_cache_minimax_h3(
                item,
                {
                    "varlen_mmh3_hidden_states_bfloat16": hidden.to(torch.bfloat16).cpu(),
                    "varlen_mmh3_token_tags_int64": tags.cpu(),
                },
                metadata,
            )


if __name__ == "__main__":
    main()
