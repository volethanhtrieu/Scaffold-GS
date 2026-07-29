# Viettel AI Race scene pipeline

The supplied data already has the competition layout:

```text
data/<scene>/
├── train/
│   ├── images/
│   └── sparse/0/{cameras.bin,images.bin,points3D.bin}
└── test/test_poses.csv
```

No data copy or move is needed. `scene/dataset_readers.py` now ignores
COLMAP records whose image is not present in `train/images`; this is required
because the supplied sparse model also contains the withheld test views.
`train.sh` also resolves a scene's nested `train/` directory automatically for
the repository's existing single-scene commands.

Run the environment verifier and read-only seven-scene audit first:

```bash
scripts/verify_sogs_environment.sh --gpu 0 --data-root data
```

On a primary GPU machine, train every scene sequentially (this command starts
training; it is intentionally not run on the auxiliary machine):

```bash
python train_competition.py --gpu 0
```

The launcher uses all available training images by passing
`data/<scene>/train` directly to `train.py`, saves models under
`outputs/round1/<scene>/`, and skips the repository's held-out evaluation
pass. Use `--dry-run` to inspect commands without starting a job.

SOGS remains an explicit method choice; it does not alter the data or
submission pipeline. Prepare (but do not run during repository migration) an
opt-in competition command such as:

```bash
python train_competition.py --gpu 0 \
  --use-second-order True --feat-dim 16 \
  --num-eigenvectors 2 --lambda-sgl 0.01 --dry-run
```

Keep the generated `cfg_args`, `sogs_config.json`, checkpoint, and training
log with each experiment so the organizer's reproducibility requirements can
be met.

Render the CSV test poses from the saved checkpoints:

```bash
python render_test_poses.py --gpu 0
```

This writes PNG predictions to `predictions/round1/<scene>/`. Finally, validate
dimensions and create the required archive:

```bash
python generate_submission.py
```

The default archive is `submission_round1.zip` with members named
`<scene>/<image_name stem>.png`, matching the PDF's PNG archive example. The
CSV values in this public data use `.JPG`/`.jpg` source names; if the organizer
explicitly requires the literal extension, use
`--filename-mode image-name` instead.

If the submission portal enforces a 350 MB archive limit, lossless PNG may be
too large for all full-resolution views. The supplied CSV names use JPEG
extensions, so a size-limited literal-name JPEG archive can be generated
without resizing any prediction:

```bash
python generate_submission.py \
  --filename-mode image-name \
  --transcode-jpeg \
  --jpeg-quality 92 \
  --jpeg-subsampling 420 \
  --max-size-mb 350
```

The output is replaced atomically only when every expected image is valid and
the completed archive is within the configured limit. If quality 92 exceeds
the limit, lower `--jpeg-quality` gradually; if it is comfortably below the
limit, try a higher value to preserve more image quality.

None of these scripts delete source data. `train_competition.py` refuses to
reuse a non-empty output directory unless `--allow-existing-output` is passed.
