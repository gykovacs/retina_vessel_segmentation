# Numerical identification of methodological inconsistencies affecting the evaluations in 100+ papers on retinal vessel segmentation, and a new baseline for the field

## Authors

* György Kovács
* Attila Fazekas

## Scope

We check the consistency of evaluation scores in the field of retinal vessel segmentation, and also provide an improved ranking by adjusting for various methodological flaws in the evaluations. This repository contains the raw data, the implementation of the analysis and all results derived.

## Contents

1. Raw data:
    * [`data/retinal_vessel_segmentation.xlsx`](data/retinal_vessel_segmentation.xlsx): the raw performance scores at the image and aggregated levels with further descriptors of the papers.
    * [`data/drive/`](data/drive): directory where the DRIVE database needs to be placed.
2. Python core:
    * [`core.py`](core.py): implementation of some core functionalities used across the notebooks.
    * [`config.py`](config.py): some configurational parameters.
3. Notebooks:
    * [`00_extract_image_statistics.ipynb`](00_extract_image_statistics.ipynb): extract the image level statistics from the images of the DRIVE database (overwrites `data/drive_stats.csv`).
    * [`01_discovery.ipynb`](01_discovery.ipynb): some basic discovery of the extracted image statistics.
    * [`02_image_level.ipynb`](02_image_level.ipynb): implementation of the image level consistency checks and analysis, the resulting Latex tables and figures are saved to `output/latex/` and `output/figures/`, respectively, the numerical results are saved to `results_with_image_level_data.csv`.
    * [`03_aggregated.ipynb`](03_aggregated.ipynb): implementation of the image level consistency checks and analysis, the resulting Latex tables and figures are saved to `output/latex/` and `output/figures/`, respectively, the numerical results are saved to `results_with_aggregated_data.csv`.
    * [`04_improved ranking.ipynb`](04_improved_ranking.ipynb): implementation of the adjustments of the scores, the resulting Latex tables and figures are saved to `output/latex/` and `output/figures/`, respectively, the numerical results are saved to `results_with_adjusted_scores.csv`.
    * [`05_the analysis.ipynb`](05_the_analysis.ipynb): generates the insights based on `results_with_adjusted_scores.csv`.
    * [`06_fov_illustration.ipynb`](06_fov_illustration.ipynb): generates the figures on the specificities of the consistency tests and saves them to `output/figures/`.
    * [`07_orchestration.ipynb`](07_orchestration.ipynb): runs all the notebooks in the right order to generate all results.
4. Output:
    * [`/output/figures/`](output/figures): all figures generated for the paper.
    * [`/output/latex/`](output/latex): all Latex tables generated for the paper.
    * [`/output/drive_stats.csv`](output/drive_stats.csv): the image level statistics extracted from the DRIVE database.
    * [`/output/results_with_image_level_data.csv`](output/results_with_image_level_data.csv): the contents of the first sheet of `data/retinal_vessel_segmentation.xlsx` extended with the results of the image level consistency tests.
    * [`/output/results_with_aggregated_data.csv`](output/results_with_aggregated_data.csv): the contents of `output/results_with_image_level_data.csv` extended with the results of the aggregated consistency tests.
    * [`/output/results_with_adjusted_scores.csv`](output/results_with_adjusted_scores.csv): the contents of `output/results_with_aggregated_data.csv` extended with the adjusted scores.
    * All other output, like the insights are within the notebooks.

## Reproducing the results

1. Download the DRIVE database and extract it to the folder `/data/drive/` (this step can be skipped, and then the notebook `00_extract_image_statistics.ipynb` won't be executed, the rest of the analysis will use the already extracted statistics from `output/drive_stats.csv`)
2. Create a new Python environment and install all dependencies. Using Anaconda, in the root folder of the repository:
```bash
> conda create -n 'retinal_vessel_segmentation' python==3.7
> conda activate retinal_vessel_segmentation
> pip install -r ./requirements.txt
```

3. Start a Jupyter notebook server
```bash
> jupyter notebook
```

4. Load and execute the notebooks.

## Remarks

1. The contents of `config.py` can be changed in order to control some parameters of the evaluations, the choices of the default values are explained in the paper.

