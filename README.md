# Parameter Inference via TVB Simulations and SBI

This project enables the simulation of brain activity from human connectomes using **The Virtual Brain (TVB)**, with models such as **Zerlaut** and specific tools like **NuuTools**. It also performs parameter inference through Simulation-Based Inference (SBI).

---

## Project Structure

Below is an overview of the main folders and files:

### `data_tvb/`
- `connectome_TVB/` : Connectomes formatted for TVB simulations.
- `real_connectome_for_tvb/` : Real connectome datasets used as input.
- `kernel/` : Precomputed kernels for models (e.g., functional connectivity).

### `generated_data/`
- Generated data files and intermediate outputs used for training or simulation.

### `results/`
- Final simulation and inference results.
  - `synch/` : Specific synchronization or other indicator measures.

### `save_file/`
- Intermediate saved files during processing.

### `simulation_file/`
- Python scripts related to launching TVB simulations and handling connectomes.

### `inference/`
- Core inference scripts and functions:
  - `sbi_simulation.py` — Main entry point script to launch SBI inference rounds.
  - `inference_core.py` — Main functions for single and multi-round inference (`single_round_inference`, `multi_round_inference`).
  - `simulator.py` — Simulator functions like `simulator_BOLD`, `simulator_FC`, and observation generation.
  - `utils.py` — Utility functions for saving/loading data, logging, matrix conversions, and simulations.
  - `prior.py` — Prior definitions and management.

### `config/`
- Configuration files containing parameters, bounds, and settings.

---

## How to Run

### 1. Clone the repository

"""
git clone <repository-url>
cd <repository-name>
"""

### 2. Create and activate a Python virtual environment

"""
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate.bat      # Windows
"""

### 3. Install dependencies

"""
pip install -r requirements.txt
"""

### 4. Create the project folder structure (if needed)

"""
python scripts/setup_structure.py
"""

### 5. Prepare the data

- Place the original dataset in `original_dataset/`.
- Use scripts in `generated_data/` to convert connectomes to a TVB-compatible format.
- Check that the necessary files exist in `data_tvb/connectome_TVB/` and `data_tvb/kernel/`.

---

## Main Script and Key Parameters

- The main script to launch inference is located at: inference/sbi_simulation.py

- This script uses core functions from `inference/inference_core.py` for running single or multi-round inference.

- Simulator functions and auxiliary utilities are in `inference/simulator.py` and `inference/utils.py`.

- Important parameters and hyperparameters to modify for inference are located in the `params` module or configuration files (e.g., `config/default_params.json`).

- The folder `simulation_file/` contains everything needed for running TVB simulations based on connectomes.

---

## Summary of Key Components

| Folder/File                     | Description                                           |
|---------------------------------|-------------------------------------------------------|
| `inference/sbi_simulation.py`   | Main entry point for SBI inference runs               |
| `inference/inference_core.py`   | Core inference logic: single and multi-round methods  |
| `inference/simulator.py`        | Simulator functions (BOLD, FC simulations, etc.)      |
| `inference/utils.py`            | Utilities for simulations and data handling           |
| `params.py`                     | Parameter and configuration files                     |
| `simulation_file/`              | Scripts for TVB simulations and connectome management |

---



