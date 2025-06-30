# Critical Slowing Down in Free Recall

This repository contains relevant code, data and documentation associated with the study "Adaptive proximity to criticality underlies amplification of ultra-slow fluctuations during free recall". The study explores how the phenomenon of "critical slowing down" may be related to the generative process involved in memory recall. More specifically, this work simulates random recurrent networks demonstrating that a small modulation towards a critical transition may lead to specific amplification in the power of slow fluctuations.

## Overview

- 📁 `code/` – Analysis and preprocessing scripts
- 📁 `data/` – Raw and processed datasets, with accompanying metadata
- 📁 `examples/` – Demonstrations for getting started
- 📁 `notebooks/` – Exploratory notebooks for model development and hypothesis testing
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
The `demos/` folder includes self-contained examples on: (a) how to run a short simulation of the random recurrent network while modulating its proximity to criticality, and (b) how to analyze the results.  



## Citation
```
@article{yellin2025critical,
  title={Adaptive proximity to criticality underlies amplification of ultra-slow fluctuations during free recall},
  authors={Yellin, Dovi and Siegel, Noam and Malach, Rafael and Shriki, Oren},
  journal={biorxiv},
  year={2025},
  doi={10.1101/2023.02.24.529043}
}
```