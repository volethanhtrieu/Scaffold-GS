# Push the SOGS Branch and Clone It on the Training Machine

This guide moves only source code and documentation through Git. The dataset,
trained models, rendered predictions, ZIP submissions, credentials, and local
paper are intentionally excluded.

Repository details at the time this guide was prepared:

```text
local branch: main
baseline:     9718569d385c618f551242402a26a7db34259c56
remote:       https://github.com/volethanhtrieu/Scaffold-GS
new branch:   feature/sogs-integration
```

No commit or push has been performed by Codex. Run the Git write commands below
in your own terminal, where `.git` and your GitHub credentials are available.

## 1. Open the modified repository

```bash
cd /home/trieu_kernel/Desktop/Viettel_drone/Scaffold-GS
git branch --show-current
git rev-parse HEAD
git status --short
```

Before creating the branch, the first two commands should show `main` and:

```text
9718569d385c618f551242402a26a7db34259c56
```

The status output should contain the SOGS migration files. Do not discard,
reset, or stash them.

## 2. Confirm that large local artifacts are ignored

```bash
git check-ignore -v \
  data \
  outputs \
  predictions/ \
  submission_round1.zip
```

Each path should be matched by `.gitignore`. This prevents the dataset,
checkpoints, rendered images, and submission archive from being staged by
`git add -A`.

## 3. Create the new branch

```bash
git switch -c feature/sogs-integration
git branch --show-current
```

The second command must print:

```text
feature/sogs-integration
```

If Git reports that the branch already exists, do not create another branch;
switch to the existing one:

```bash
git switch feature/sogs-integration
```

## 4. Review the unstaged migration

```bash
git status --short
git diff --check
git diff --stat
```

`git diff --check` should print nothing and exit successfully. Review the
status before staging. In particular, confirm that it does not list dataset
images, `outputs/`, `predictions/`, a ZIP file, a credential file, or an
environment secret.

## 5. Stage and inspect the exact commit

The worktree was clean before this migration, so stage the current
non-ignored migration changes:

```bash
git add -A
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
```

Do not continue if the cached file list contains data, trained results,
predictions, ZIP archives, API tokens, passwords, SSH keys, or unrelated
personal files. Unstage an accidental path without deleting it:

```bash
git restore --staged path/that/should/not/be/committed
```

Then add the correct files and inspect the cached list again.

## 6. Create the commit

```bash
git commit -m "Add optional SOGS integration and competition pipeline"
git log -1 --oneline
git status --short
```

The final status should be clean. If it is not, inspect the remaining files
before deciding whether they belong in a second commit.

## 7. Push the new branch to GitHub

The configured remote is named `origin`. Confirm it before pushing:

```bash
git remote -v
```

Push the branch and configure its upstream:

```bash
git push -u origin feature/sogs-integration
```

For an HTTPS remote, authenticate with your GitHub credential helper or
personal access token when prompted. Do not place a token directly in the
command or commit it to a file.

Verify that the remote branch exists:

```bash
git ls-remote --heads origin feature/sogs-integration
git status --short
```

`git ls-remote` should print a commit hash followed by:

```text
refs/heads/feature/sogs-integration
```

## 8. Clone that branch on the training machine

On the training machine, choose a parent directory and run:

```bash
git clone \
  --branch feature/sogs-integration \
  --single-branch \
  https://github.com/volethanhtrieu/Scaffold-GS.git \
  Scaffold-GS-SOGS

cd Scaffold-GS-SOGS
git branch --show-current
git log -1 --oneline
```

The branch must be `feature/sogs-integration`, and the last commit should be
the commit created in Step 6.

The CUDA extension sources under `submodules/` are ordinary tracked
directories in this repository, not Git submodules. A normal clone includes
them; no `git submodule update` command is required.

## 9. Create or activate the training environment

If this machine does not yet have the environment:

```bash
conda env create -f environment.yml
conda activate scaffold_gs
```

If it already has a compatible `scaffold_gs` environment, activate it without
recreating it:

```bash
conda activate scaffold_gs
```

Run the strict environment check before transferring or training data:

```bash
scripts/verify_sogs_environment.sh --gpu 0
```

The script checks required Python imports, a small CUDA tensor operation, both
compiled extensions, Python and shell syntax, and the SOGS synthetic tests. It
does not start training. Fix every `[FAIL]` before continuing. Optional
packages may be reported separately without blocking competition training.

## 10. Transfer or mount the dataset separately

The `data/` directory is ignored by Git, so cloning the branch does not copy
the seven scenes. Copy, mount, or synchronize the dataset separately until the
training clone has:

```text
data/HCM0421/train/
data/HCM0421/test/
data/HCM0539/train/
data/HCM0539/test/
data/HCM0540/train/
data/HCM0540/test/
data/HCM0644/train/
data/HCM0644/test/
data/HCM0674/train/
data/HCM0674/test/
data/bonsai/train/
data/bonsai/test/
data/chair/train/
data/chair/test/
```

After transferring the data, rerun the verifier with the data audit enabled:

```bash
scripts/verify_sogs_environment.sh \
  --gpu 0 \
  --data-root data
```

Do not start training unless the summary says:

```text
Environment verification PASSED. No training was started.
```

and the data audit says:

```text
Result: 7/7 scene(s) valid.
```

## 11. Start the documented seven-scene workflow

Continue at Step 4 of
[`docs/sogs_run_guide.md`](sogs_run_guide.md). First execute its dry-run
command, inspect all seven resolved commands, and only then run the real
training command.

## 12. Pull later branch updates safely

If more commits are pushed to the same branch, update the training clone with:

```bash
git status --short
git pull --ff-only origin feature/sogs-integration
```

Run the environment verifier again after pulling. `--ff-only` prevents Git
from creating an unexpected merge commit on the training machine.
