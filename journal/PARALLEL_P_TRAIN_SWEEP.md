# Independent execution of the fixed p_train stage

This helper accelerates the already declared 672-case `p_train_sweep` while the running main controller completes confirmation and the m sweep. It adds no model, seed, hyperparameter candidate or test-based selection. The main controller, code checkout, status file and manifests remain untouched.

`run_parallel_p_train_sweep.py` requires the complete main `manifests/screen.jsonl` and frozen `validation_selection_results.json`. It invokes the original builder with those exact absolute input paths. Tests verify that its JSONL and plan are byte-identical to the main controller's future output. All scientific and evaluation settings remain fixed: three prescribed seeds, four algorithms, four datasets, fourteen outage probabilities, CPU/two-thread execution and ten-draw full-risk evaluation.

The external execution checkout must be clean and committed. Committed and working-tree blobs for `run_experiment.py`, `theorem1.py`, `cifar_models.py`, `cifar_search_models.py`, `campaign.py`, `build_bayesian_campaign.py` and `build_campaign.py` must match the running main revision `f602987`. The external coordinator itself must match its own committed blob. Training metadata records the external commit, and the publication audit attests its training/builder equivalence to the main commit.

Before submitting computation, probe the actual destination filesystem from the prepared external checkout:

```bash
conda run -n torch28 python -m journal.run_parallel_p_train_sweep \
  --probe-publication /home/yangshuo/Git/bnn_bayesian_refined_20261117/results/journal_bayesian
```

The probe creates and removes an owned temporary directory under the supplied parent. It requires a successful whole-directory move and `EEXIST` without modification for both existing empty and nonempty destinations. Only Linux `renameat2(RENAME_NOREPLACE)` is permitted. Unsupported systems/filesystems fail before expensive computation; plain `os.rename` is never a fallback. The runtime repeats the probe and requires staging, destination and probe parent to have the same filesystem device.

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

Publication moves the **entire completed stage directory** to `MAIN_ROOT/p_train_sweep` with one atomic absent-only rename. The helper never copies individual trials into an active main folder. A read of the main status requires an advisory confirmation/m-sweep window; this read does not provide race protection. `RENAME_NOREPLACE` provides that protection: if the main controller creates its stage first—even an empty directory—the move fails with `EEXIST`, preserving both directories. If publication wins first, the main controller later sees a complete cache before starting any trial.

Before the move, `publication_snapshot/` retains the external global JSON/CSV files and exact manifests/selection inputs. These preserve original absolute path strings because a later main resume legitimately rewrites its global row paths. `publication_audit.json` records original and target paths, every task's path map and saved source fingerprints, manifest hashes, source blobs, resource job ID, and the pre-move directory device/inode. The main resume path has a synthetic saved-checkpoint regression proving cached training/evaluation reuse without subprocess execution, new returned paths, unchanged saved bytes and accepted frozen-source bounds.

An existing complete external stage is validated and reused without rewriting global results, preserving its frozen publication archive. A failed publication leaves the complete stage external. A signal after the kernel commits the move is identified from the directory's recorded device/inode and is never reported as unpublished; a later same-source resume reconciles only the external audit/status, even if the main has already advanced. No rollback or active-main repair is attempted.

The helper has not been launched by its implementation or tests. Execution and publication status come from its own `status.json`, logs and audit. If the publication window closes or another stage already owns the destination, the helper fails closed and retains its external evidence for review.
