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

Run the read-only audit first:

```bash
python prepare_data.py
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

None of these scripts delete source data. `train_competition.py` refuses to
reuse a non-empty output directory unless `--allow-existing-output` is passed.
