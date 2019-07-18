## Code for "Microarray data analysis in prediction of breast cancer metastasis"

Directory and file contents:

* [data/](data/) folder contains only markdown file with brief decription of the data set as the original files are too big to upload them to github.
* [mlcc_results/](mlcc_results/) folder contains results of MLCC executed on our data (clustering of variables, clusters' dimensionalities).
* [notebook_results/](notebook_results) folder contains computation demanding notebooks executed using [papermill](https://papermill.readthedocs.io/en/latest/#).
* [notebooks/](notebooks/) folder contains all notebooks used to generate reuslts from the thesis:
    * [Exploration](notebooks/Exploration.ipynb) and [Normalizations](notebooks/Normalizations.ipynb) include code used in section 3 of the thesis.
    * [DataPreparation](notebooks/DataPreparation.ipynb) includes code splitting our data to train and test sets.
    * [functions](notebooks/functions.ipynb) contains all imports and definitions of used models as well as all helper methods (plotting ROC curves, printing scores, wrappers for sklearn models, wrappers for models executed in R, etc.).
    * [Regularized](notebooks/Regularized.ipynb) and [DimensionalityReduction](notebooks/DimensionalityReduction.ipynb) include code for fine tuning models from section 2.2 and section 2.3 of the thesis.
    * [ModelSelection](notebooks/ModelSelection.ipynb) performs nested corss-validation for algorithm selection
    * [RandomLogisticRegression](notebooks/RandomLogisticRegression.ipynb) includes code for fine tuning and nested cross-validation of RLR model only. It was separated from other models due to long execution time.
    * [ThresholdAdjustment](notebooks/ThresholdAdjustment.ipynb) includes code for adjusting decision threshold and presentation of precision-recall tradeoff from section 4.2 of the thesis
    * [SAM](notebooks/SAM.ipynb) performs Significance analysis of microarrays in R (section 4.3).
    * [ComputeSVCFeatureRanking](notebooks/ComputeSVCFeatureRanking.ipynb) contains code for Recursive feature elimination using SVC classifier (section 4.3).
    * [FeatureSelection](notebooks/FeatureSelection.ipynb) includes code for comparison of feature selection methods from section 4.3 of the thesis.
    * [miscellaneous](notebooks/miscellaneous.ipynb) contains some helper code and a part not used in the thesis.
* [selection_results/](selection_results/) contains results of SAM and RFE.
* [config.yaml](config.yaml) contains a configuration for running notebooks using papermill by [run_notebooks.py](run_notebooks.py) script.
* [environment.yaml](environment.yaml) contains exported conda environment in which the code was executed.
* [mlcc_run_script.R](mlcc_run_script.R) is a script running MLCC on our train set.

The thesis itself can be found in the [thesis repository](https://github.com/sjwilczynski/thesis).