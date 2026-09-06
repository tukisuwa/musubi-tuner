# MiniMax-H3 indexed multi-frame inference (MFI)

This branch adds an experimental, opt-in H3 layout for generating several independent image roles in one DiT call.

MFI uses one shared coordinate contract:

```text
clean visual conditions: {(latent, signed pixel-frame index)}
noisy target roles:      {(latent, signed pixel-frame index)}
model output:             target rows only
```

The indices are MM-RoPE coordinates, not built-in semantic labels. MFI does not prescribe a role set, role count, tensor order, or index assignment. Those meanings are learned from the paired targets and are defined by the training contract together with the stable tensor order, indices, reference route, and noise policy.

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
  --prompt "generate the jointly trained output roles" \
  --lora_weight /path/to/multi_role_lora.safetensors \
  --width 512 --height 512 --steps 20 --seed 1 \
  --h3_independent_target_roles \
  --h3_target_frame_indices=-3,2 \
  --h3_visual_condition_frame_indices=0 \
  --h3_target_noise_coupling independent \
  --output_type latent_images --output outputs/mfi
```

The PNG names preserve slot and index; with the example above they are `000_index_-3.png` and `001_index_+2.png`. `--frame_count` is not used to size an MFI target; the number of target indices determines its latent length.

`shared` target noise broadcasts the first target slice's initial noise to every role. It preserves each slice's standard-normal marginal but changes cross-role covariance. Models trained with `independent` noise should normally be inferred with `independent` noise.

## Training

The training cache must expose:

- target video latents shaped `[B,24,N,H,W]`, with `N` equal to the number of target indices;
- a positive-length audio placeholder and `audio_present=0` for image-only training;
- reference latent rows and Qwen vision rows consistent with the selected reference route.

Example training flags for two joint roles are:

```bash
accelerate launch -m musubi_tuner.minimax_h3_train_network \
  --task ref2va --dataset_config /path/to/dataset.toml \
  --dit /path/to/ref2va_dit.safetensors \
  --video_only --mixed_precision bf16 \
  --network_module networks.lora_minimax_h3 --network_dim 8 --network_alpha 8 \
  --h3_independent_target_roles \
  --h3_target_frame_indices=-3,2 \
  --h3_visual_condition_frame_indices=0 \
  --h3_target_noise_coupling independent \
  --h3_reference_route dual
```

Independent role caches can be assembled by encoding each role as a T=1 image, verifying that their reference latents are identical, and concatenating their target latents along latent time in the same order as `--h3_target_frame_indices`. Keep this derivation in cache metadata; the trainer records the active MFI contract in LoRA metadata.

### Dataset and cache workflow

`python -m musubi_tuner.minimax_h3_cache_mfi` builds these caches directly. It accepts either a JSONL manifest, a directory of grouped images (`name_0000.png`, `name_0001.png`, ...), or a directory of videos. Targets and controls are encoded independently as single images; sparse source frames must not be concatenated and encoded as an ordinary H3 video.

For example, `dataset.jsonl` can contain this record (relative paths resolve from the manifest directory):

```json
{"id":"example","caption":"a character turning around","relative":true,"controls":[{"path":"reference.png","index":10,"mask":"mask.png"}],"targets":[{"path":"before.png","index":4},{"path":"after.png","index":18}]}
```

The cached target coordinates are `[-6,8]` and the control coordinate is `0`. Another record may have different coordinates and a different number of images. A source entry with `"path":"clip.mp4","frame":24,"index":8` selects frame 24 after resampling the source at 24 fps and assigns model coordinate 8. Source frame selection and model coordinates are separate.

Run the two stages in separate processes so the Qwen and VAE models do not stay resident together:

```bash
python -m musubi_tuner.minimax_h3_cache_mfi \
  --manifest dataset.jsonl --output cache/mfi --stage latents \
  --video_vae /path/to/video_vae.safetensors \
  --audio_vae /path/to/audio_vae.safetensors --route dual

python -m musubi_tuner.minimax_h3_cache_mfi \
  --manifest dataset.jsonl --output cache/mfi --stage text \
  --text_encoder /path/to/qwen3-vl.safetensors \
  --text_encoder_blocks_to_swap 50 --route dual
```

Both stages must use the same source selection, dimensions, seed, relative-positioning settings and reference route. Paired cache fingerprints are checked before model loading. Existing cache files are never overwritten; use a new directory when changing a contract. The tool records source file identities and model fingerprints.

Instead of `--manifest`, use:

- `--image_directory images --control_indices 2 --target_indices 0 4`: groups files by the prefix before their numeric suffix. Captions come from `<prefix>.txt`, falling back to `--caption`. Without explicit selection, the center image is the control and the remaining images are targets.
- `--video_directory videos --num_controls 2 --max_targets 4 --max_frame_distance 32 --samples_per_video 3`: deterministically samples controls and nearby targets using `--seed`. Each draw samples the target count uniformly from 1 through the smaller of `max_targets` and the available candidate count, then samples that many indices. Selection happens at caching time. Video captions come from the adjacent `.txt` file. Explicit `--control_indices` and `--target_indices` override random selection. This count distribution changes selections relative to caches made before this behavior was added; rebuild both stages together in a new cache directory.
- `--relative`: subtracts the smallest control index from both sets. Each manifest record can override this with `relative`.

Use the resulting caches with a standard image dataset, for example:

```toml
[general]
resolution = [512, 512]
batch_size = 1

[[datasets]]
image_directory = "/path/to/source_images"
cache_directory = "/path/to/cache/mfi"
```

Train with `--h3_independent_target_roles --video_only --h3_reference_route dual --task ref2va`. Omit the two index CLI options to use each sample's cached indices. If CLI indices are provided, they must match the cache; a conflicting override raises an error. The current H3 trainer uses batch size 1, so variable target/control counts can coexist in a run. Gradient accumulation remains available.

Training-time samples also support MFI. Specify `--h3_target_frame_indices -6,8 --h3_visual_condition_frame_indices 0` in each sample-prompt line (or use the run-wide flags). Each sampled target is saved as its own PNG. Cache-derived indices alone cannot choose the intended layout for a separate sample prompt, so sampling requires explicit coordinates.

### Relative coordinates, interpolation and output order

`--h3_relative_positioning` subtracts the smallest explicit visual-condition index from both target and condition indices. With it omitted, the supplied indices are used directly. This coordinate conversion does not change Qwen's text positions; H3 does not promise invariance when only visual coordinates are translated relative to text/audio.

`--h3_interpolate --h3_interpolation_thresholds 2 3` adds target indices between the union of requested targets and controls. The first threshold applies before the earliest control, the second after it. Pure control positions are not generated; a position listed in both sets remains both a target and a condition. All added targets participate in the same denoising call. This changes the joint attention context and noise shape, so the original targets can change too.

By default, only requested targets are decoded, in the requested order. `--h3_save_interpolated` enables interpolation and saves all generated indices in ascending order. The saved latent retains the full sampling layout and the output-slot map, allowing a separate `--latent_path ... --output_type images` invocation to recover the same images without repeating the MFI flags.

`--h3_sequence_video` additionally writes `sequence.mp4` in coordinate order at 24 fps. Missing indices are represented by holding the preceding image; they are not estimated motion. Use interpolation with thresholds `1 1` and save the added targets when those intermediate positions should actually be generated. Clean control images are not inserted into the output automatically.

### Masks and initial target state

Repeat `--h3_control_mask mask.png` once per visual condition to mask its normalized VAE latent. White preserves the condition, black sets it to zero; masks are area-resized to the latent grid. Dataset control entries support the same `mask` field. This masks only DiT conditioning: Qwen still sees the complete control image, and this is not an attention mask or a target loss mask.

Targets normally start from independent Gaussian noise. `--h3_initial_image image.png` broadcasts one encoded source image to all target slices; repeat it once per expanded target to provide different sources. Alternatively, `--h3_initial_latent source.safetensors` reads a `latent_video` tensor of the exact expanded target shape. The source is mixed with the selected noise according to `--h3_strength`:

- `1`: pure noise start, the default;
- `0`: preserve the source latent;
- values between 0 and 1: start at a truncated H3 noise schedule and denoise the source.

The video and audio clocks both start at the selected base sigma, using their respective H3 shifts. Strength is a base-sigma fraction, so it is not the linear video noise mixing coefficient after the video shift. Image sources use VAE posterior-mode encoding. Training keeps the usual flow-matching objective; using a source at inference does not imply that the LoRA was trained on that source distribution. To study a source-conditioned training contract, define and evaluate that contract separately.

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

Generation accepts the same `--h3_reference_route` option (also per prompt in `--from_file`). Training-time samples automatically inherit the training route. Caption-only routes remove reference rows from the Qwen presentation; routes without DiT references omit those tokens entirely. `text_only` may use just `--prompt`, without reference inputs. Explicit control indices may still define relative target positioning, but are not inserted as DiT reference tokens on routes without DiT references. Existing text caches must match the routed presentation; incompatible caches are rejected.

FL2VA MFI training samples accept first-only, last-only, or both controls, with one explicit visual-condition index per provided image. Batch MFI outputs omit placeholder audio, and decode-only also ignores placeholder audio in older MFI files.

## Compatibility and limits

- The feature is opt-in. With the MFI flags omitted, native H3 video, one-frame, ConvRot/NVFP4, and long-duration behavior is unchanged.
- MFI is separate from the official `--one_frame` mode and temporal stretching.
- Explicit visual indices support image Ref2VA references (up to the H3 limit of nine) and FL2VA image conditions. Video dataset frames enter MFI as independently encoded images.
- Joint roles attend to one another. Joint generation is therefore not equivalent to running one role at a time, even with matched noise slices.
- Preserve target count, tensor order, indices, route, and noise coupling in every reproducibility manifest.
