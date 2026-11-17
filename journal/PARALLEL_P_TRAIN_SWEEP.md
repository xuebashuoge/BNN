# Independent execution of the fixed p_train stage

This helper accelerates the already declared 672-case `p_train_sweep` while the running main controller completes confirmation and the m sweep. It adds no model, seed, hyperparameter candidate or test-based selection. The main controller, code checkout, status file and manifests remain untouched.

`run_parallel_p_train_sweep.py` requires the complete main `manifests/screen.jsonl` and frozen `validation_selection_results.json`. It invokes the original builder with those exact absolute input paths. Tests verify that its JSONL and plan are byte-identical to the main controller's future output. All scientific and evaluation settings remain fixed: three prescribed seeds, four algorithms, four datasets, fourteen outage probabilities, CPU/two-thread execution and ten-draw full-risk evaluation.

The external execution checkout must be clean and committed. Committed and working-tree blobs for `run_experiment.py`, `theorem1.py`, `cifar_models.py`, `cifar_search_models.py`, `campaign.py`, `build_bayesian_campaign.py` and `build_campaign.py` must match the running main revision `f602987`. The external coordinator itself must match its own committed blob. Training metadata records the external commit, and the publication audit attests its training/builder equivalence to the main commit.

Before submitting computation, probe the actual destination filesystem from the prepared external checkout:

```bash
conda run -n torch28 python -m journal.run_parallel_p_train_sweep \
  --probe-publication /home/yangshuo/Git/bnn_bayesian_refined_20261117/results/journal_bayesian
```

The probe creates and removes an owned temporary directory under the supplied parent. It requires successful atomic directory symlink creation to an absolute, persistent source, and `EEXIST` without modification for existing empty directories, nonempty directories, files and broken symlinks. Only `os.symlink(..., target_is_directory=True)` is permitted. Unsupported filesystems fail before expensive computation; directory renaming is never a fallback. The runtime repeats the probe and requires staging, destination and probe parent to have the same filesystem device. The earlier GPFS `renameat2` probe returned `EINVAL`; no acceleration job was launched under that rejected contract.

After the frozen screening selection exists, submit from the external committed checkout with `torch28` and the shared `DATASET` configured:

```bash
export JOURNAL_GIT_SHA="$(git rev-parse HEAD)"
export PARALLEL_MAIN_ROOT=/home/yangshuo/Git/bnn_bayesian_refined_20261117/results/journal_bayesian/20261117_refined
export PARALLEL_MAIN_REVISION=f602987
export PARALLEL_OUTPUT=/home/yangshuo/Git/bnn_bayesian_parallel_ptr_20261117/results/parallel_ptr/20261117
sbatch journal/parallel_p_train_sweep.slurm
```

The Slurm allocation requests 32 CPUs, 32 GiB and 90 minutes, using 16 workers with two threads each. Before computation, the helper checks `2 GiB + 1.25 * workers * main_pilot_max_worker_RSS` against the allocated memory. It refuses an existing main `p_train_sweep` directory, including an empty directory or symbolic link. Its owned output is protected by an exclusive lock and immutable run protocol.

All training and evaluation occur under the external output's `stage/`. Publication requires exact coverage of every one of the 672 training/test identities and configurations, all three seeds, complete histories and tensor integrity, full ten-draw risk/decomposition reports, saved bank/split verification, and matching metadata/weights/evaluation fingerprints. Symbolic links are refused throughout the completed stage. Source blobs, frozen selection inputs, manifest and validated source bytes are checked again before publication. The provisional bound calculation used during integrity checking is discarded; the main controller computes final bounds using its complete-family confidence allocation.

Publication creates `MAIN_ROOT/p_train_sweep` as an **atomic directory symlink to the entire completed external stage**. The source stage stays at its original absolute path permanently and must not be deleted or cleaned after the job. The helper never copies individual trials into an active main folder. A read of the main status requires an advisory confirmation/m-sweep window; this read does not provide race protection. Atomic symlink creation provides that protection: if the main controller creates its stage first—even an empty directory or broken link—publication fails with `EEXIST`, preserving the source and existing target. If publication wins first, the main controller's `mkdir(exist_ok=True)` follows the valid directory link, and every worker sees a complete cache.

Before the link is created, `publication_snapshot/` retains the external global JSON/CSV files and exact manifests/selection inputs. Main resume legitimately rewrites aggregate status/row files through the link; their original bytes remain archived outside the stage. Per-checkpoint metadata, weights, histories and full evaluations remain unchanged. `publication_audit.json` records original and published paths, every task's path map and saved source fingerprints, manifest hashes, source blobs, resource job ID, publication kind and the persistent source directory's device/inode. Original report checkpoint paths remain valid because the source stays alive. A synthetic saved-checkpoint regression runs the actual `campaign.main --resume` entry point through the link with training/inference subprocesses forbidden, checks that global rows are written through the link while scientific bytes remain unchanged, and verifies that frozen-source bound observation accepts the result.

An existing complete unpublished external stage is validated and reused without rewriting global results, preserving its frozen publication archive. After publication the external helper never trains or evaluates again. Recovery requires a symlink with the exact recorded absolute source path and the same source directory device/inode. A signal after the kernel commits the link is never reported as unpublished; a later same-source resume reconciles only the external audit/status, even if the main has already advanced. No unlink, source move, rollback or active-main repair is attempted. The main controller may only reuse the fixed scientific checkpoints and rewrite aggregate files through this published cache.

The helper has not been launched by its implementation or tests. Execution and publication status come from its own `status.json`, logs and audit. If the publication window closes or another stage already owns the destination, the helper fails closed and retains its external evidence for review.
