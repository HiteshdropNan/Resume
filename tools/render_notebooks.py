"""
Automated Notebook Pre-Rendering Script
=======================================
Executes the specified Jupyter notebooks using nbconvert ExecutePreprocessor
and saves the embedded outputs (matplotlib plots, tables, stdout) in place so
that GitHub renders the graphics immediately for viewers.
"""

import os
import sys
from pathlib import Path

# Prevent permission issues with default user config paths
repo_root = Path(__file__).resolve().parent.parent
tmp_mpl = repo_root / ".tmp_mpl"
tmp_mpl.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(tmp_mpl)

tmp_ipython = repo_root / ".tmp_ipython"
tmp_ipython.mkdir(exist_ok=True)
os.environ["IPYTHONDIR"] = str(tmp_ipython)

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = [
    "01_algorithmic_trading_cqf/algorithmic_trading_walkthrough.ipynb",
    "07_deep_pde_fbsde_derivative_pricing/deep_fbsde_pde_solver.ipynb",
    "08_credit_risk_cds_bootstrapping/cds_hazard_rate_bootstrapping.ipynb",
    "09_vecm_cointegration_trading/vecm_johansen_cointegration.ipynb",
    "10_black_litterman_portfolio/black_litterman_allocation.ipynb",
]

def main():
    repo_root = Path(__file__).resolve().parent.parent
    print("=" * 70)
    print("Pre-Rendering Jupyter Notebook Outputs for GitHub Viewer")
    print(f"Repository Root: {repo_root}")
    print("=" * 70)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    for rel_path in NOTEBOOKS:
        full_path = repo_root / rel_path
        if not full_path.exists():
            print(f"ERROR: Notebook not found: {full_path}")
            continue

        print(f"\n[Executing] {rel_path} ...", flush=True)
        working_dir = str(full_path.parent)

        with open(full_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        try:
            ep.preprocess(nb, {"metadata": {"path": working_dir}})
            with open(full_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            with_out = [c for c in code_cells if c.outputs]
            print(f"  ==> Success! {len(with_out)}/{len(code_cells)} code cells populated with visual outputs.", flush=True)
        except Exception as e:
            print(f"  ==> ERROR during execution: {e}", flush=True)
            raise e

    print("\n" + "=" * 70)
    print("All target notebooks have been pre-rendered and saved with visual charts!")
    print("=" * 70)


if __name__ == "__main__":
    main()
