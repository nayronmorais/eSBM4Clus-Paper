<h2> Benchmark data, results, and all source codes used in the paper "An Evolving Approach to the Similarity-Based Modeling for Online
Clustering in Non-Stationary Environments" </h2>

Acessing Experimental Data
------------------------------
- The scripts (jupyter notebooks) used to create the figures illustrating the approach features of the approach, its hyperparameters rationale and the datasets are available in `notebooks/ilustrations`.
- The datasets used to compare the performance of the methods are available in the folder `datasets`.
- The results presented in the experimental part of the paper are available in the folder `formatted results`.
- The scripts used in the experiments are avaliable in the root directory and in the folder `extra scripts`.

How to Run 
------------
-   **Install Packages:**
    
    -   Install the package `evos` from [https://github.com/nayronmorais/evos](https://github.com/nayronmorais/evos) along with its dependencies.
    -   Install the [pympler](https://pympler.readthedocs.io/) and [pandas](https://pandas.pydata.org) libraries.
-   **Run the Random Search Evaluation:**
    
    -   Run the file `main.py` to perform the random search evaluation. The default settings are 100 runs per method for each dataset, using Adjusted Rand Index (ARI) as the scoring metric, with CPU parallelization over 30 processes. This will create a folder named `results` containing all the results.
-   **Summarize Random Search Results:**
    
    -   Run the file `extra scripts/summarize_search_results.py`. This will create a folder named `formatted results` containing a CSV file named `search_results.csv` with the best results from the random search for all methods and all datasets.
-   **Evaluate Computational Burden and Purity:**
    
    -   Run the file `extra scripts/eval_computational_burden_and_purity.py` to perform the computational burden evaluation (memory consumption and time spent per sample) for each method, and also compute the Purity performance using the best hyperparameter settings obtained from the random search. This will create a CSV file named `purity_results.csv` inside `formatted results`, and also a subfolder named `computational burden` containing the results for this evaluation.
-   **Generate Computational Burden Graphs:**
    
    -   Run the file `make_graphs_computational_burden.py` to build the graphs related to the computational evaluation. The generated graphs will be located in `formatted results/computational burden/graphs/`.
-   **Generate Hyperparameters Stability Graphs:**
    
    -   Run the file `make_graphs_hyper_stability.py` to build the graphs related to the hyperparameters stability evaluation. The generated graphs will be located in `formatted results/computational burden/varying_hyperparameters/`.
