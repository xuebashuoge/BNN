# Recovering historical working copies

The project reached its file-count quota on 19 November 2026 during the harder fading and erasure comparisons. The completed earlier bank-size study was archived without changing any numerical records or Git references.

- `completed_mgrid_scientific_runs.tar` contains the three complete historical raw directories m200, m500 and m1000 from `diagonal_fading_mgrid_20261119`. The project archive was compared byte-for-byte against the source before migration. The archive passed a second comparison against all migrated files and a SHA-256 verification.
- The original project paths for those directories remain symbolic links to persistent working copies under `/home/yangshuo/Git/bnn_task_inode_recovery_20261119/mgrid_working_copy`. The previously published figures, tables, repository commit and deliverable archive remain in their original locations. The project tar is the group-accessible preservation copy.
- `pinned_completed_worktrees.tar` preserves ten completed detached run checkouts. The logical worktree paths remain usable through links to the user's persistent Spartan home. Four smaller completed QA worktrees are also retained under that home, with a verified archive and full working copies. Research result directories from running jobs were excluded from these migrations.
- To inspect a working tree without the home links, extract the project tar in an empty directory with sufficient fileset quota. The original directory names and file bytes are preserved.
- Current incomplete runs that encountered the quota were staged separately for inspection. Recovery retained good evaluations and usable weights and retried remaining task indices using the same manifests, seeds, numerical settings and method definitions.

The migration scripts and their verification logs are in the same Spartan home directory as the retained working copies.

For additional headroom, the completed earlier study's logs, small-sample moons runs, original moons comparison and MNIST bank-size sweep were also archived independently in `legacy_completed_support.tar`. The archive was compared with its source before moving the active copies into `earlier_completed_support` in the same persistent Spartan home directory, then compared again after the move. Their original paths are symlinks. `completed_qa_worktrees.tar` stores the four smaller completed QA checkouts on project storage as well.
