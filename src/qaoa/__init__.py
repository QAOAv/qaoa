# src/qaoa/__init__.py

from .qaoa_general import (
    QAOAConfig,
    QAOAResult,
    run_qaoa,
    run_qaoa_sweep,
    # Hamiltonian builders for easy student access:
    maxcut_hamiltonian,
    maxcut_cost_fn,
    number_partition_hamiltonian,
    number_partition_cost_fn
)

__version__ = "0.1.0"
__author__ = "Kumar & Mattaparthi"