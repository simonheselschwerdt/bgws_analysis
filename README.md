# **Blue-Green Water Share Analysis**

## **Title**
**Large impact of extreme 5-day precipitation on projected blue-green water shares**

---

## Table of Contents

- [Description](#description)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Authors and Acknowledgment](#authors-and-acknowledgment)
- [License](#license)

## Description
This repository contains code to reproduce the findings of the work in preliminary manuscript 'Large impact of extreme precipitation on projected blue-green water shares' (by Simon P. Heselschwerdt*, Yannis Markonis, Anna M. Ukkola, Thorsten Wagener, Lan Wang-Erlandsson, Yuting Yang, and Peter Greve), and this README describes the code and data for reproducibility.

### **Notebook-Centric Workflow**
The project uses Jupyter notebooks for all major tasks, including data ingestion, preprocessing, analysis, and visualization. The `src` directory stores reusable functions and utilities to support the notebooks, ensuring modular and clean code.

### **Code availability**
We provide all code to reproduce the main results and all figures of the paper. Either browse/download files individually or clone the repository to your local machine (see installation). 

### **Data availability** 
Due to storage and copyright constraints, original data have to be downloaded from their sources (given in Data Availability Statement).

## **Demo**  
For demonstration purposes, users can run the analysis using the default settings provided in `config.py`. These settings define the data directory, model selection, variable names, and other parameters required to execute the scripts. Simply run the Jupyter Notebooks with the default settings to load a demo dataset and run some of the code.

This will allow you to reproduce only parts of the analysis (e.g. historical maps of precipitation for default model). To fully reproduce all results, please follow the instructions in the Jupyter Notebooks and refer to the Methods section of the manuscript.  

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository:**
   ```plaintext
   git clone https://github.com/simonheselschwerdt/bgws_analysis.git
   cd bgws_analysis
   ```
2. **Set up a Conda environment:**
   ```plaintext
   conda env create -f environment.yml
   conda activate bgws_analysis
   ```
3. **Verify installation: Ensure all dependencies are installed by running:**
   ```plaintext
   python --version
   jupyter --version
   ```
## Usage of notebooks

1. **Download Data**
   - *Define your data directory in the configuration file (src/config.py) first.*
   - Use the following notebook to download the required datasets:
     ```plaintext
     notebooks/import_and_preprocess/import_CMIP6_data_intake.ipynb
     ```
3. **Preprocess Data**
   - Preprocessing includes tasks like setting common timestamps, regridding, applying landmask, removing Antarctica and Greenland/Iceland, and converting units:
     ```plaintext
     notebooks/import_and_preprocess/preprocess_CMIP6_data.ipynb
     ```
4. **Compute Variables**
   - Compute derived variables such as WUE, VPD, evaporation, and RX5day:
     ```plaintext
     notebooks/import_and_preprocess/compute_variables_indices.ipynb
     ```
5. **Generate Maps**
   - Create global maps of the blue-green water share and ensemble statistics:
     ```plaintext 
     notebooks/analysis_and_plots/global_maps.ipynb
     ```
6. **Generate Parallel Coordinate Plots**
   - Compute spatial means and generate parallel coordinate plots for the analysis:
     ```plaintext
     notebooks/analysis_and_plots/parallel_coordinate_plots.ipynb
     ```
8. **Perform Regression Analysis and Plot Importance**
   - Perform regression analysis and plot importance scores:
     ```plaintext
     notebooks/analysis_and_plots/regression_analysis_and_plots.ipynb
     ```

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
│   ├── __init__.py                               # Package initialization
│   ├── config.py                                 # Configuration file for defining directories and parameters
│   ├── data_handling/
│   │   ├── __init__.py 
│   │   ├── compute_statistics.py        
│   │   ├── load_data.py
│   │   ├── process_data.py
│   │   └── save_data_as_nc.py
│   ├── analysis/        
│   │   ├── __init__.py 
│   │   ├── regression_analysis.py
│   │   └── regression_analysis_rf.py
│   └── visualization/ 
│   │   ├── __init__.py 
│   │   ├── colormaps.py        
│   │   ├── parallel_coordinate_plots.py
│   │   └── regression_analysis_results.py
├── environment.yml                               # Conda environment specification
├── LICENSE                                       # License file
└── README.md                                     # Project documentation
```            

## Authors and acknowledgment
The Github repository is maintained by the corresponding author (Simon P. Heselschwerdt, Email: [simon.heselschwerdt@hereon.de](mailto:simon.heselschwerdt@hereon.de)). 
All acknowledgements and references will be available in the published paper.

## License
This project is licensed under the [MIT License](LICENSE).
