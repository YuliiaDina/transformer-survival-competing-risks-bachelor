# Transformer-based Survival Analysis with Competing Risks

This repository contains the official implementation of the code for the Bachelor Thesis at "Харківський Політехнічний Інститут"
The project explores the application of Deep Learning, specifically **Transformer architectures**, to Survival Analysis in the presence of Competing Risks. It compares novel attention-based neural networks (Standard and Monotonic Attention) against classical statistical baselines like Cox Proportional Hazards and Fine-Gray Regression.


## Quick Start 

The easiest way to explore the models and view the results is through Google Colab. The notebook is configured to automatically clone this repository, install all dependencies, and run the complete pipeline.

1. Click the **Open in Colab** badge at the top of this Readme.
2. In the Google Colab menu, select `Runtime` -> `Run all`.
3. The notebook will automatically process the data, train the baselines and Transformers, and generate calibration plots and metrics tables.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuliiaDina/transformer-survival-competing-risks-bachelor/blob/main/main_notebook.ipynb)

## Local Development Setup

If you wish to run the code locally or contribute to the repository, this project uses [Poetry](https://python-poetry.org/) for strict dependency management.

### Prerequisites
- Python 3.12+
- Poetry installed (`pip install poetry`)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YuliiaDina/transformer-survival-competing-risks-bachelor.git](https://github.com/YuliiaDina/transformer-survival-competing-risks-bachelor.git)
   cd transformer-survival-competing-risks-bachelor
2. Install dependencies:
   ```bash
   poetry install
3. Run the analysis pipeline (or open the Jupyter Notebook locally):
   ```bash
   poetry run jupyter notebook
