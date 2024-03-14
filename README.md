# Analysis of Differential Privacy and Synthetic Data: Balancing Data Utility, Anonymization Effectiveness, and Energy Consumption

This repository contains the code associated with the relevant research paper. It enables the reproduction of the report and its results. Please be advised that executing all the code may take more than a day.

## Prerequisites

Before running the scripts, ensure you have satisfied the following prerequisites:

1. **Install Required Python Packages**

   Execute the following command to install the necessary packages listed in `requirements.txt`:

   ```bash
   pip3 install -r requirements.txt
   ```

2. **Modify Permissions for Power Consumption Data Access**

   For pyRAPL measurements, modify permissions by running:

   ```bash
   sudo chmod -R a+r /sys/class/powercap/intel-rapl
   ```

   Note: This pyRAPL configuration should be re-executed at every system restart, when using this repository.

3. **Good To Know**

    When generating datasets and in the results, an **epsilon-value of 0 denotes synthetic data**. If the **epsilon-value is not 0, it's a differentialy private dataset**.

## Scripts Overview

The repository includes eight scripts, each serving a specific purpose in the research process:

1. **00-prepare_dir_and_data.sh**
   
   - Sets up the all necessary directories and downloads the two datasets

2. **01-idle.py**
   
   - Measures the idle system energy usage to establish a baseline for further energy consumption analysis
   - Results are stored in `energy/idle.csv`

3. **02-preprocessing.py**
   
   - Splits the datasets into a training set and a control/validation set. It also performs some minor cleaning to prepare the data for analysis
   - The final datasets are stored in `data/{placeholder}/{placeholder}_train.csv` and `data/{placeholder}/{placeholder}_val.csv`

4. **03-dataset_generation.py**
   
   - Measures the energy consumption involved in generating datasets, both synthetic and using Differential Privacy (DP)
   - Results are stored in `energy/dataset_generation.csv`
   - Temporary files are stored in the directory: `dataset_generation`
   - Note: Synthetic datasets have the `epsilon` of 0

5. **04-dp.py**
   
   - Measures the system's energy usage when handling DP Query-Response operations
   - Results are stored in `energy/dp.csv`

6. **05-anonymization-effectiveness.py**
   
   - Assesses the reidentification risks for synthetic (SYN) datasets and DP datasets across a range of epsilon values from 0 to 20
   - Results and temporary files are stored in `anonymization_effectiveness`where the results are stored in the file `risks_{placeholder}.csv`
   - Note: Synthetic datasets have the `epsilon` of 0
   - Note: Risks are invalidated when the `main`  attack has a lower succes than `naive` (guessing) attack

7. **06-data-utility.py**
   
   - Evaluates data utility using k-NN and Logistic Regression (LogReg). The tests include benchmarks based solely on the original dataset, as well as datasets generated from synthetic data and DP.
   - Results and temporary files are stored in `data_utility`, with results being the file `results.csv`
   - Note: Synthetic datasets have the `epsilon` of 0

8. **07-aggregate_results.py**
   
   - Compiles all the generated results from the previous scripts and presents them in an organized manner.
   - The output graphs are stored in the directory `results_output`


   
