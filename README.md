# Critical Slowing Down in Free Recall

This repository contains relevant code, data and documentation associated with the study "Adaptive proximity to criticality underlies amplification of ultra-slow fluctuations during free recall". The study explores how the phenomenon of "critical slowing down" (CSD) may be related to the generative process involved in memory recall. More specifically, this work simulates random recurrent networks demonstrating that a small modulation towards a critical transition may lead to specific amplification in the power of slow fluctuations, as observed in the empirical study of free recall using iEEG.

## Overview

- 📁 `csd/` – Simulation infrastructure and analysis codebase
- 📁 `data/` – Raw and processed datasets, with accompanying metadata
- 📁 `examples/` – Getting started demonstrations
- 📁 `figures/` – Scripts for analysis and generation of figures 
- 📁 `notebooks/` – Scripts demonstrating network analysis
- 📁 `results/` – Statistical outputs and visualizations used in the publication
- 📁 `docs/` – Supplementary materials and manuscript source files


## How to install
Clone this repository and install dependencies:

```bash
git clone https://github.com/dovi-yellin/critical_slowing_down_in_free_recall.git
cd critical_slowing_down_in_free_recall
```

We recommend using a virtual environment:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Getting Started
The `examples/` folder includes self-contained examples on: (a) how to run a short simulation of the random recurrent network while modulating its proximity to criticality, and (b) how to analyze the results.  

To run this complete workflow, start at the project root and execute python `demo_run_simulation.py`: the script loads simulation parameters from `test_general.json`, lets you override any of them directly in the code (e.g., change network size or run-time), simulates the rate-model across a small parameter sweep, and writes the outputs to `results/rate_model_*.pkl`. Note the size of the generated `pkl` is larger than 5 GB. 

When simulation finishes, launch python `demo_run_analysis.py`; this second script automatically reads the freshly generated `.pkl`, extracts the activity traces and provides the ability to visualize results (e.g., plot power-spectral-density curves for each run etc.).



## Citation
```
@article{yellin2025critical,
  title={Adaptive proximity to criticality underlies amplification of ultra-slow fluctuations during free recall},
  authors={Yellin, Dovi and Siegel, Noam and Malach, Rafael and Shriki, Oren},
  journal={PLOS Computational Biology},
  year={2025},
  doi={https://doi.org/10.1371/journal.pcbi.1013528}
}
```
