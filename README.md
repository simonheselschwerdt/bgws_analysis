# **bgws-analysis**

## **Publication Name**
**Large impact of extreme precipitation on global blue-green water share under climate change**

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Authors and Acknowledgment](#authors-and-acknowledgment)
- [License](#license)
- [Contact](#contact)

## Description
This repository contains code to reproduce the findings of the work in preliminary manuscript 'Large impact of extreme precipitation on global blue-green water share under climate change' (by ...), and this README describes the code and data for reproducibility.

Code availability: We provide all code to reproduce the main results and all figures of the paper. Either browse/download files individually or clone the repository to your local machine (git clone https://github.com/simonheselschwerdt/bgws_analysis.git). 

Data availability: Due to storage and copyright constraints, original CMIP6 data have to be downloaded from their original sources (given in Data Availability Statement).

## Features

- Scripts to download and preprocess CMIP6 data.
- Reproducible workflows for statistical analysis.
- Code to generate all figures in the paper.
- Jupyter notebooks for data visualization and exploration.

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository:**
   git clone https://github.com/simonheselschwerdt/bgws_analysis.git
   cd bgws_analysis

2. **Set up a Conda environment:**
   conda env create -f environment.yml
   conda activate bgws_analysis

3. **Verify installation: Ensure all dependencies are installed by running:**
   python --version
   jupyter --version

## Usage of notebooks

1. **Download Data**
   *Define your data directory in the configuration file (src/config.py) first.*
   Use the notebook to download the required datasets:
   notebooks/import_and_preprocess/import_CMIP6_data_intake.ipynb
3. **Preprocess Data**
   Preprocessing includes: set common timestamps, regrid, apply landmask, remove Antartica and Greenland/Iceland and convert units.
   notebooks/import_and_preprocess/preprocess_CMIP6_data.ipynb
4. **Compute Variables**
   Compute variables WUE, VPD, evaporation and RX5day.
   notebooks/import_and_preprocess/compute_variables_indices.ipynb
5. **Generate Maps**
   Here, we compute the bgws and ensemble stats.
   notebooks/analysis_and_plots/global_maps.ipynb
6. **Generate Parallel Coordinate Plots**
   Here, we additionally compute spatial means.
   notebooks/analysis_and_plots/parallel_coordinate_plots.ipynb
8. **Perform Regression Analysis and Plot Permutation Importance**
   Here, we perform the regression analysis and get the importance scores.
   notebooks/analysis_and_plots/regression_analysis_and_plots.ipynb

## Directory Structure

```plaintext
bgws-analysis/
├── notebooks/                                      
│   │   ├── import_CMIP6_data_intake.ipynb            
│   │   ├── preprocess_CMIP6_data.ipynb
│   │   └── compute_variables_indices.ipynb
│   ├── analysis_and_plots/     
│   │   ├── global_maps.ipynb
│   │   ├── parallel_coordinate_plots.ipynb
│   │   └── regression_analysis_and_plots.ipynb
├── src/                                            
│   ├── __init__.py
│   ├── config.py
│   ├── data_handling/
│   │   ├── __init__.py 
│   │   ├── compute_statistics.py        
│   │   ├── load_data.py
│   │   ├── process_data.py
│   │   └── save_data_as_nc.py
│   ├── analysis/        
│   │   ├── __init__.py 
│   │   └── regression_analysis.py
│   └── visualization/ 
│   │   ├── __init__.py 
│   │   ├── colormaps.py        
│   │   ├── parallel_coordinate_plots.py
│   │   └── regression_analysis_results.py
├── environment.yml        
├── LICENSE                
└── README.md
```            

## Authors and acknowledgment
The Github repository is maintained by the corresponding author (Simon P. Heselschwerdt, Email: [simon.heselschwerdt@hereon.de](mailto:simon.heselschwerdt@hereon.de)). 
All acknowledgements and references will be available in the published paper.

## License
This project is licensed under the [MIT License](LICENSE).
