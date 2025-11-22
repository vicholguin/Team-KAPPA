CSCI 5415 Data Mining – Group KAPPA
Phase 6: Project Results

OVERVIEW

This repository contains the final, refactored pipeline code for the group project. The goal is to provide a complete, reusable, and reproducible end-to-end data mining workflow that:

Loads data from two different application domains.

Preprocesses and engineers features.

Trains and evaluates multiple classification models.

Mines and analyzes association rules.

Generates figures and saves them to disk.

Can be run top-to-bottom without manual intervention.

The main script is designed to handle both datasets using a common pipeline structure with clearly defined functions for each step.

FILES

Phase6_finalpipeline_TeamKAPPA.main()
Main Python script implementing the full pipeline. It includes:

Automatic package checks and installation (using pip) at runtime.

Data loading functions for each dataset via kagglehub.

Dataset-specific preprocessing functions.

Model training and evaluation pipeline for multiple classifiers.

Association rule mining pipeline using Apriori and FP-Growth.

Automatic figure saving (no interactive plots).

requirements.txt
List of Python packages required to run the pipeline.

(Generated at runtime)
plots/
global/
PNG figures for the Global Superstore dataset.
ecommerce/
PNG figures for the E-Commerce Consumer Behaviour dataset.

PYTHON AND ENVIRONMENT

Recommended Python version: 3.10 or later.

The script uses scikit-learn, xgboost, imbalanced-learn, mlxtend, kagglehub, and common scientific Python libraries.

Even though the script will attempt to auto-install missing packages at runtime, it is strongly recommended to create a virtual environment and install requirements explicitly:

python -m venv venv
source venv/bin/activate (Linux/Mac)
venv\Scripts\activate (Windows)

pip install -r requirements.txt

DATASETS

The pipeline works with two datasets from distinct application domains:

Global Superstore dataset

Source: Kaggle (“Global Super Store Dataset” by apoorvaappz)

Domain: Retail, order-level data.

Used to define and predict high-value customers (binary classification) and to mine product association rules.

E-Commerce Consumer Behaviour 2023 dataset

Source: Kaggle (“ECommerce Dataset for Predictive Marketing 2023” by hunter0007)

Domain: Online retail and consumer behaviour.

Used to predict product reordering (binary classification) and to mine association rules on product co-occurrence.

The script uses kagglehub to download both datasets. You must have Kaggle API credentials configured on your system so kagglehub can authenticate and access the datasets.

If kagglehub is not available or you prefer local files, you can modify the load_data() function in finalmodel_pipeline.py to read from local CSV paths instead of kagglehub.dataset_download().

PIPELINE STRUCTURE

The pipeline is organized into stages, each with its own function:

Data loading

load_data(dataset_name)

Supported dataset_name values:

"global": Global Superstore dataset

"ecommerce": E-Commerce Consumer Behaviour dataset

Uses kagglehub to download the dataset and returns:

df (main working DataFrame)

df_copy (copy reserved for association rules and additional analysis)

Preprocessing and feature engineering

preprocess_dataset(df, dataset_name)

For "global":

Drops non-predictive columns (e.g., postal code).

Aggregates profit per Customer ID and creates a High_Value binary target:
High_Value = 1 for top 25% of customers by total profit, 0 otherwise.

Performs basic outlier capping on selected numeric features.

Extracts time-based features from Order Date and Ship Date (year, month).

For "ecommerce":

Handles missing values (e.g., days_since_prior_order).

Caps numeric outliers for selected order-related columns.

Ensures reordered is a 0/1 integer binary target.

For classification, a generic helper (preprocess_global_binary) is used:

Drops user-specified non-predictive columns.

Label-encodes categorical features.

Imputes numeric values with median.

Scales numeric features with StandardScaler.

Applies SMOTE to handle class imbalance on the training data.

Splits into training and test sets.

Modeling (classification)

train_models_for_dataset(df, dataset_name)

Calls run_pipeline_for_all_models(), which:

Preprocesses data for classification.

Trains and tunes four classifiers:

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

XGBoost

Uses GridSearchCV or RandomizedSearchCV with StratifiedKFold cross-validation for hyperparameter tuning.

Evaluates each model on the held-out test set.

For each model, it computes:

Accuracy

ROC-AUC

Precision-Recall AUC (PR-AUC)

F1 Score

Balanced Accuracy

Baseline balanced accuracy (majority-class baseline)

It returns:

results_df: a table summarizing best parameters and metrics for all models.

models_dict: a dictionary of the best fitted models.

prep: preprocessing artifacts (imputer, scaler, encoders, etc.).

Evaluation and plotting

evaluate_and_plot(model, X_test_scaled, y_test, model_name, feature_names, ...)

For each classifier, it:

Prints classification report and metrics.

Generates and saves:

Confusion matrix heatmap.

ROC curve.

Precision-Recall curve.

Feature importance bar chart (for tree-based models).

All figures are saved automatically using save_fig(), which writes PNG files to:
plots/<dataset_name>/...

Association rules

run_association_rules_for_dataset(df_copy, dataset_name)

Calls rule_based_pipeline() with dataset-specific settings.

Steps:

Builds a transactional basket (order x product) matrix via make_basket().

Mines frequent itemsets using:

Apriori (apriori from mlxtend.frequent_patterns).

FP-Growth (fpgrowth from mlxtend.frequent_patterns).

Generates association rules with association_rules().

Computes support count and string representations of antecedent/consequent.

Prunes rules using thresholds on support, confidence, lift, and maximum rule length.

Produces a final pruned rules DataFrame.

It also creates and saves:

Top-K rules by lift (horizontal bar chart).

Support vs confidence scatter plot.

Top-level pipeline control

run_pipeline_for_dataset(dataset_name)

Runs the complete sequence for a single dataset:

load_data()

preprocess_dataset()

EDA summary and basic plots

train_models_for_dataset()

run_association_rules_for_dataset()

Returns a dictionary with:

summary (feature summary DataFrame)

results_df (model metrics)

models (best models)

prep (preprocessing assets)

rules_result (association-rule outputs)

main()

Calls run_pipeline_for_dataset("global").

Calls run_pipeline_for_dataset("ecommerce").

Prints completion message.

HOW TO RUN

Ensure Python 3.10+ is installed.

Create and activate a virtual environment (recommended).

Install dependencies:

pip install -r requirements.txt

Make sure your Kaggle API credentials are correctly configured on your system so kagglehub can download the datasets. Refer to Kaggle’s API documentation for details on setting up kaggle.json.

Run the main script from the command line:

python finalmodel_pipeline.py

The script will:

Download both datasets using kagglehub.

Preprocess each dataset.

Train and evaluate all classification models.

Mine association rules for each dataset.

Save all plots as PNG files in the plots/ directory.

Print key metrics and rule summaries to the console.

OUTPUTS

After successful execution, you should see:

Console output:

Progress messages for loading, preprocessing, training, and mining.

Model metrics (accuracy, ROC-AUC, PR-AUC, F1, balanced accuracy).

Final rule counts and example rules for each dataset.

Files:

plots/global/*.png (EDA, classification evaluation, association rule visuals for Global Superstore)

plots/ecommerce/*.png (same for E-Commerce dataset)

NOTES AND LIMITATIONS

kagglehub is used for convenience; if the environment does not have Kaggle support, load_data() can be modified to read local CSV copies of the datasets.

The script includes a simple auto-install mechanism for required packages. In some managed or restricted environments, this may not work; in that case, install dependencies manually using requirements.txt.

Hyperparameter grids and sampling settings are chosen for a balance between runtime and performance. They can be adjusted for faster experimentation or more exhaustive search.