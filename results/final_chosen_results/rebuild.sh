#!/usr/bin/env bash
# Run from the torch28 environment.
set -euo pipefail
cd "$(dirname "$0")"
python draw_mnist_analog_finite_channel_generalization.py
python draw_cifar10_channel_mismatch.py
python plot_bound_decomposition_d8.py
python rebuild_performance_table.py
python rebuild_final_tables.py
