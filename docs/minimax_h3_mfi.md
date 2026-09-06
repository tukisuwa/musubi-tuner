# MiniMax-H3 indexed multi-frame inference (MFI)

This branch adds an experimental, opt-in H3 layout for generating several independent image roles in one DiT call. It is intended for tasks such as producing `highlight`, `base`, and `shadow` layers from one reference image.

MFI uses one shared coordinate contract:

```text
clean visual conditions: {(latent, signed pixel-frame index)}
noisy target roles:      {(latent, signed pixel-frame index)}
model output:             target rows only
```

The indices are MM-RoPE coordinates, not built-in semantic labels. `-1` does not intrinsically mean highlight. Role meaning is learned from the paired targets, stable tensor order, indices, reference route, and noise policy together.

## Generation

Use `--h3_independent_target_roles` with one target index per output. MFI generation accepts `latent`, `images`, or `latent_images`; image modes decode every target slice independently through the VAE's temporal-padded T=1 path.

```bash
python minimax_h3_generate_video.py \
  --task ref2va \
  --dit /path/to/ref2va_dit.safetensors \
  --video_vae /path/to/video_vae.safetensors \
  --audio_vae /path/to/audio_vae.safetensors \
  --text_encoder /path/to/qwen3-vl \
  --ref /path/to/reference.png \
  --prompt "separate the illustration into editable lighting layers" \
  --lora_weight /path/to/three_layer_lora.safetensors \
  --width 512 --height 512 --steps 20 --seed 1 \
  --h3_independent_target_roles \
  --h3_target_frame_indices=-1,0,1 \
  --h3_visual_condition_frame_indices=0 \
  --h3_target_noise_coupling independent \
  --output_type latent_images --output outputs/mfi
```

The PNG names preserve slot and index, for example `000_index_-1.png`, `001_index_+0.png`, and `002_index_+1.png`. `--frame_count` is not used to size an MFI target; the number of target indices determines its latent length.

`shared` target noise broadcasts the first target slice's initial noise to every role. It preserves each slice's standard-normal marginal but changes cross-role covariance. Models trained with `independent` noise should normally be inferred with `independent` noise.

## Training

The training cache must expose:

- target video latents shaped `[B,24,N,H,W]`, with `N` equal to the number of target indices;
- a positive-length audio placeholder and `audio_present=0` for image-only training;
- reference latent rows and Qwen vision rows consistent with the selected reference route.

Example training flags for three joint roles are:

```bash
accelerate launch -m musubi_tuner.minimax_h3_train_network \
  --task ref2va --dataset_config /path/to/dataset.toml \
  --dit /path/to/ref2va_dit.safetensors \
  --video_only --mixed_precision bf16 \
  --network_module networks.lora_minimax_h3 --network_dim 8 --network_alpha 8 \
  --h3_independent_target_roles \
  --h3_target_frame_indices=-1,0,1 \
  --h3_visual_condition_frame_indices=0 \
  --h3_target_noise_coupling independent \
  --h3_reference_route dual
```

Independent role caches can be assembled by encoding each role as a T=1 image, verifying that their reference latents are identical, and concatenating their target latents along latent time in the same order as `--h3_target_frame_indices`. Keep this derivation in cache metadata; the trainer records the active MFI contract in LoRA metadata.

## Reference routes

`--h3_reference_route` validates the actual cached inputs instead of simulating a missing route with zero tensors.

| Route | Qwen image rows | DiT reference latent |
| --- | --- | --- |
| `dual` | yes | yes |
| `qwen_image_only` | yes | no |
| `dit_latent_only` | no (caption-only text cache) | yes |
| `text_only` | no (caption-only text cache) | no |
| `native` | follows `--task` | follows `--task` |

The route experiments still use the Ref2VA base family (`--task ref2va`). For `qwen_image_only` and `text_only`, the latent cache has a T2VA packed layout; for `dit_latent_only` and `text_only`, the Qwen cache must contain no vision rows.

## Compatibility and limits

- The feature is opt-in. With the MFI flags omitted, native H3 video, one-frame, ConvRot/NVFP4, and long-duration behavior is unchanged.
- MFI is separate from the official `--one_frame` mode and temporal stretching.
- Explicit visual indices currently support image-only Ref2VA references.
- Joint roles attend to one another. Joint generation is therefore not equivalent to running one role at a time, even with matched noise slices.
- Preserve target count, tensor order, indices, route, and noise coupling in every reproducibility manifest.
