"""
CSCI 5415 Data Mining - Group KAPPA
Phase 6: Project Results

This script refactors the original notebook into a reusable pipeline with
clear stages:

    1. load_data(dataset_name)
    2. preprocess_dataset(df, dataset_name)
    3. train_models_for_dataset(df, dataset_name)
    4. run_association_rules_for_dataset(df_copy, dataset_name)
    5. run_pipeline_for_dataset(dataset_name)
    6. main(): runs pipeline for both 'global' and 'ecommerce'

Datasets:
    - 'global'    → Global Superstore
    - 'ecommerce' → E-Commerce Consumer Behaviour
"""
# ==========================================================
#  AUTO-INSTALL REQUIRED PACKAGES IF MISSING
# ==========================================================
import importlib
import subprocess
import sys

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",
    "imbalanced-learn",
    "mlxtend",
    "kagglehub",
]

def install_if_missing(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"\n'{package}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅  Installed '{package}' successfully.\n")

for pkg in REQUIRED_PACKAGES:
    install_if_missing(pkg)



# =======================
#  Basic Python & System
# =======================
import os
import time
import warnings

import numpy as np
import pandas as pd

# =======================
#  Visualization
# =======================
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
#  Preprocessing
# =======================
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)

# =======================
#  Models
# =======================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =======================
#  Evaluation
# =======================
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# =======================
#  Imbalanced Data
# =======================
from imblearn.over_sampling import SMOTE

# =======================
#  Association Rules
# =======================
from mlxtend.frequent_patterns import (
    apriori,
    fpgrowth,
    association_rules,
)

warnings.filterwarnings("ignore", category=Warning)
pd.set_option("display.width", 500)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", None)

# =======================
#  Data Source Config
# =======================
try:
    import kagglehub
except ImportError:
    kagglehub = None
    print(
        "WARNING: kagglehub not installed. "
        "Install it or replace load_data() with local file paths."
    )

DATA_CONFIG = {
    "global": {
        "kaggle_id": "apoorvaappz/global-super-store-dataset",
        "filename": "Global_Superstore2.csv",
        "encoding": "latin1",
        "target_col": "High_Value",
        "drop_cols": [
            "Row ID",
            "Order ID",
            "Order Date",
            "Ship Date",
            "Customer ID",
            "Product ID",
        ],
    },
    "ecommerce": {
        "kaggle_id": "hunter0007/ecommerce-dataset-for-predictive-marketing-2023",
        "filename": "ECommerce_consumer behaviour.csv",
        "encoding": "latin1",
        "target_col": "reordered",
        "drop_cols": [
            "order_id",
            "user_id",
            "product_id",
            "days_since_prior_order",
            "department_id",
        ],
    },
}

# =============================================================================
#  Generic EDA helpers (reused from original)
# =============================================================================


def data_preprocessing_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a feature-wise statistics DataFrame (type, missing rate, mean, etc.)
    Used as EDA summary, not required for modeling.
    """
    stats_dict = {
        "Feature": [],
        "Type": [],
        "Non-Null Count": [],
        "Missing Count": [],
        "Missing Rate": [],
        "Unique Values": [],
        "Mean": [],
        "Median": [],
        "Std": [],
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": [],
        "Mode": [],
        "Skewness": [],
    }

    for feature in data.columns:
        col = data[feature]
        stats_dict["Feature"].append(feature)
        stats_dict["Type"].append(col.dtype)
        stats_dict["Non-Null Count"].append(col.shape[0] - col.isnull().sum())
        stats_dict["Missing Count"].append(col.isnull().sum())
        stats_dict["Missing Rate"].append(col.isnull().sum() / col.shape[0])
        stats_dict["Unique Values"].append(col.nunique())

        if col.dtype in ["float64", "int64"]:
            stats_dict["Mean"].append(col.mean())
            stats_dict["Median"].append(col.median())
            stats_dict["Std"].append(col.std())
            stats_dict["Min"].append(col.min())
            stats_dict["25%"].append(col.quantile(0.25))
            stats_dict["50%"].append(col.median())
            stats_dict["75%"].append(col.quantile(0.75))
            stats_dict["Max"].append(col.max())
            stats_dict["Mode"].append(col.mode()[0] if not col.mode().empty else None)
            stats_dict["Skewness"].append(col.skew())
        else:
            stats_dict["Mean"].append(None)
            stats_dict["Median"].append(None)
            stats_dict["Std"].append(None)
            stats_dict["Min"].append(None)
            stats_dict["25%"].append(None)
            stats_dict["50%"].append(None)
            stats_dict["75%"].append(None)
            stats_dict["Max"].append(None)
            stats_dict["Mode"].append(col.mode()[0] if not col.mode().empty else None)
            stats_dict["Skewness"].append(None)

    return pd.DataFrame(stats_dict)


def plot_numeric_distributions(df, bins=30, figsize=(15, 10), title="Numerical Distributions"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        print("No numeric columns found in the dataframe.")
        return
    df[num_cols].hist(bins=bins, figsize=figsize)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_correlation_heatmap(df, figsize=(10, 6), cmap="coolwarm", annot=True):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        print("No numeric columns found.")
        return
    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_outliers_iqr(df, exclude_keywords=("id", "code")):
    """
    Before/after IQR capping boxplots for numeric columns (visual only).
    Does NOT modify df.
    """

    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return np.where(series > upper, upper, np.where(series < lower, lower, series))

    num_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if not any(keyword in col.lower() for keyword in exclude_keywords)
    ]

    if len(num_cols) == 0:
        print("No numeric columns found.")
        return

    plt.figure(figsize=(10, len(num_cols) * 3))

    for i, col in enumerate(num_cols, 1):
        capped_values = cap_outliers(df[col])

        plt.subplot(len(num_cols), 2, 2 * i - 1)
        sns.boxplot(x=df[col])
        plt.title(f"Original {col}")

        plt.subplot(len(num_cols), 2, 2 * i)
        sns.boxplot(x=capped_values)
        plt.title(f"Capped {col}")

    plt.tight_layout()
    plt.show()


# =============================================================================
#  Outlier helpers for numeric capping
# =============================================================================
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    q1_val = dataframe[col_name].quantile(q1)
    q3_val = dataframe[col_name].quantile(q3)
    IQR = q3_val - q1_val
    up_limit = q3_val + 1.5 * IQR
    low_limit = q1_val - 1.5 * IQR
    return low_limit, up_limit


def check_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return bool(
        dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None)
    )


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


# =============================================================================
#  Classification: preprocessing and model training
# =============================================================================
def preprocess_global_add_high_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute customer-level profit, label High_Value customers, and merge back.

    High_Value = 1 if customer's total profit is in top 25%, else 0.
    """
    df = df.copy()

    # Aggregate profit per customer
    customer_profits = df.groupby("Customer ID")["Profit"].sum().reset_index()
    customer_profits.rename(columns={"Profit": "Total_Profit"}, inplace=True)

    # Threshold for high vs low value
    threshold = customer_profits["Total_Profit"].quantile(0.75)
    customer_profits["High_Value"] = (customer_profits["Total_Profit"] >= threshold).astype(int)

    # Merge back to orders
    df = df.merge(customer_profits[["Customer ID", "High_Value"]], on="Customer ID", how="left")

    # Basic numeric outlier capping (optional but matches original intent)
    num_cols = [
        col
        for col in df.columns
        if df[col].dtype != "O"
        and col not in ["Row ID", "Order Date", "Ship Date", "High_Value"]
    ]
    for col in num_cols:
        if check_outliers(df, col):
            replace_with_thresholds(df, col)

    # Date-based features (optional; they become candidate predictors)
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
        df["order_year"] = df["Order Date"].dt.year
        df["order_month"] = df["Order Date"].dt.month

    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce", dayfirst=True)
        df["ship_year"] = df["Ship Date"].dt.year
        df["ship_month"] = df["Ship Date"].dt.month

    return df


def preprocess_ecommerce_reordered(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for ecommerce dataset, ensuring `reordered` is 0/1 and
    handling some numeric outliers.
    """
    df = df.copy()

    # Fill days_since_prior_order with median if present
    if "days_since_prior_order" in df.columns:
        df["days_since_prior_order"] = df["days_since_prior_order"].fillna(
            df["days_since_prior_order"].median()
        )

    # Outlier capping on selected numeric columns
    num_cols = ["order_number", "order_dow", "order_hour_of_day", "add_to_cart_order"]
    for col in num_cols:
        if col in df.columns:
            if check_outliers(df, col):
                replace_with_thresholds(df, col)

    # Ensure target is integer 0/1
    if "reordered" in df.columns:
        df["reordered"] = df["reordered"].fillna(0).astype(int)

    return df


def preprocess_global_binary(
    df,
    target_col,
    drop_cols=None,
    test_size=0.3,
    random_state=42,
):
    """
    Global tabular preprocessing for binary classification:
      - drop non-predictive columns
      - label-encode categoricals
      - impute numeric
      - scale numeric
      - SMOTE on training set
    """
    if drop_cols is None:
        drop_cols = []

    df_clean = df.drop(columns=drop_cols, errors="ignore").copy()

    obj_cols = df_clean.select_dtypes(include="object").columns
    le_dict = {}
    for col in obj_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        le_dict[col] = le

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col].values
    feature_names = X.columns

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    return {
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "feature_names": feature_names,
        "imputer": imputer,
        "scaler": scaler,
        "label_encoders": le_dict,
    }


def fit_with_search(
    name,
    estimator,
    search_type,
    param_grid,
    X_train_res,
    y_train_res,
    X_test_scaled,
    y_test,
    scoring="roc_auc",
    n_iter=10,
    cv_splits=3,
    random_state=42,
    verbose=1,
):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    if search_type == "grid":
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
        )
    else:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
            random_state=random_state,
        )

    start = time.time()
    search.fit(X_train_res, y_train_res)
    end = time.time()

    best_model = search.best_estimator_
    train_time = end - start

    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    majority_class = np.bincount(y_test).argmax()
    baseline_preds = np.full_like(y_test, majority_class)
    baseline_bal_acc = balanced_accuracy_score(y_test, baseline_preds)

    metrics = {
        "model": name,
        "best_params": search.best_params_,
        "cv_score": search.best_score_,
        "train_time_sec": train_time,
        "test_accuracy": acc,
        "test_roc_auc": roc,
        "test_pr_auc": pr_auc,
        "test_f1": f1,
        "test_balanced_acc": bal_acc,
        "baseline_balanced_acc": baseline_bal_acc,
    }

    print(f"\n===== {name} =====")
    print("Best Params:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {roc:.4f}")
    print(f"Test PR-AUC: {pr_auc:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Test Balanced Accuracy: {bal_acc:.4f}")
    print(f"Baseline Balanced Accuracy: {baseline_bal_acc:.4f}")

    return best_model, metrics


def evaluate_and_plot(
    model,
    X_test_scaled,
    y_test,
    model_name,
    feature_names,
    show_feature_importance=False,
    top_n=10,
):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {model_name} Metrics ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC={pr_auc:.3f}")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.show()

    if show_feature_importance and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
        print(f"\nTop {top_n} Feature Importance - {model_name}:")
        print(fi)

        plt.figure(figsize=(10, 6))
        plt.barh(fi["feature"][::-1], fi["importance"][::-1])
        plt.title(f"Feature Importance (Top {top_n}) - {model_name}")
        plt.show()


def run_pipeline_for_all_models(df, target_col, drop_cols):
    """
    Training + evaluation for KNN, Decision Tree, Random Forest, XGBoost.
    Returns: (results_df, best_models_dict, prep_dict)
    """
    prep = preprocess_global_binary(df, target_col, drop_cols)
    X_train_res = prep["X_train_res"]
    y_train_res = prep["y_train_res"]
    X_test_scaled = prep["X_test_scaled"]
    y_test = prep["y_test"]
    feature_names = prep["feature_names"]

    results = []
    best_models = {}

    # 1) KNN
    knn = KNeighborsClassifier()
    param_knn = {
        "n_neighbors": [3, 7, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    best_knn, m_knn = fit_with_search(
        name="KNN",
        estimator=knn,
        search_type="grid",
        param_grid=param_knn,
        X_train_res=X_train_res,
        y_train_res=y_train_res,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
    )
    results.append(m_knn)
    best_models["KNN"] = best_knn

    evaluate_and_plot(
        best_knn,
        X_test_scaled,
        y_test,
        model_name="KNN",
        feature_names=feature_names,
    )

    # 2) Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    param_dt = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [8, 10, 12, 14],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
    }
    best_dt, m_dt = fit_with_search(
        name="DecisionTree",
        estimator=dt,
        search_type="random",
        param_grid=param_dt,
        X_train_res=X_train_res,
        y_train_res=y_train_res,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        n_iter=40,
    )
    results.append(m_dt)
    best_models["DecisionTree"] = best_dt

    evaluate_and_plot(
        best_dt,
        X_test_scaled,
        y_test,
        model_name="Decision Tree",
        feature_names=feature_names,
    )

    # 3) Random Forest
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    param_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 6, 10, 14],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    best_rf, m_rf = fit_with_search(
        name="RandomForest",
        estimator=rf,
        search_type="random",
        param_grid=param_rf,
        X_train_res=X_train_res,
        y_train_res=y_train_res,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        n_iter=20,
    )
    results.append(m_rf)
    best_models["RandomForest"] = best_rf

    evaluate_and_plot(
        best_rf,
        X_test_scaled,
        y_test,
        model_name="Random Forest",
        feature_names=feature_names,
        show_feature_importance=True,
        top_n=15,
    )

    # 4) XGBoost
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        predictor="auto",
    )
    param_xgb = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "scale_pos_weight": [1, 2],
    }
    best_xgb, m_xgb = fit_with_search(
        name="XGBoost",
        estimator=xgb,
        search_type="random",
        param_grid=param_xgb,
        X_train_res=X_train_res,
        y_train_res=y_train_res,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        n_iter=10,
    )
    results.append(m_xgb)
    best_models["XGBoost"] = best_xgb

    evaluate_and_plot(
        best_xgb,
        X_test_scaled,
        y_test,
        model_name="XGBoost",
        feature_names=feature_names,
        top_n=15,
    )

    results_df = pd.DataFrame(results)
    return results_df, best_models, prep


# =============================================================================
#  Association Rules helpers (from original)
# =============================================================================
def make_basket(df_orders, order_col, product_col, min_item_freq=10):
    freq_by_product = (
        df_orders[[order_col, product_col]]
        .drop_duplicates()
        .groupby(product_col)[order_col]
        .nunique()
    )
    popular_products = freq_by_product[freq_by_product >= min_item_freq].index
    df_pop = df_orders[df_orders[product_col].isin(popular_products)].copy()

    basket = (
        df_pop[[order_col, product_col]]
        .drop_duplicates()
        .assign(value=1)
        .pivot_table(
            index=order_col,
            columns=product_col,
            values="value",
            fill_value=0,
        )
    )

    print(f"Popular products (>= {min_item_freq} orders): {len(popular_products)}")
    print(f"Basket shape: {basket.shape} (orders x products)")
    return df_pop, basket


def prune_rules(
    rules_df,
    min_support=0.001,
    min_confidence=0.2,
    min_lift=1.0,
    max_antecedent_len=3,
    max_consequent_len=1,
):
    if rules_df.empty:
        return rules_df

    df = rules_df.copy()
    df["ante_len"] = df["antecedents"].apply(len)
    df["cons_len"] = df["consequents"].apply(len)

    df = df[
        (df["ante_len"] <= max_antecedent_len)
        & (df["cons_len"] <= max_consequent_len)
    ]
    df = df[
        (df["support"] >= min_support)
        & (df["confidence"] >= min_confidence)
        & (df["lift"] >= min_lift)
    ]
    return df.drop(columns=["ante_len", "cons_len"])


def enrich_rules(rules_df, n_transactions: int):
    if rules_df.empty:
        return rules_df

    df = rules_df.copy()
    df["antecedent_str"] = df["antecedents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    df["consequent_str"] = df["consequents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    df["rule_len"] = df["antecedents"].apply(len) + df["consequents"].apply(len)
    df["support_count"] = (df["support"] * n_transactions).round().astype(int)
    return df


def rule_based_pipeline(
    df_orders: pd.DataFrame,
    order_col: str,
    product_col: str,
    dataset_label: str = "",
    min_item_freq: int = 100,
    min_supports=(0.01,),
    max_itemset_len: int = 2,
    sample_n_orders: int | None = None,
    prune_kwargs: dict | None = None,
):
    if prune_kwargs is None:
        prune_kwargs = dict(
            min_support=min(min_supports),
            min_confidence=0.2,
            min_lift=1.0,
            max_antecedent_len=max_itemset_len,
            max_consequent_len=1,
        )

    print(f"\n=== Rule-based pipeline for {dataset_label or 'dataset'} ===")
    print(f"Original rows: {df_orders.shape[0]}")

    if sample_n_orders is not None:
        unique_orders = df_orders[order_col].unique()
        if sample_n_orders < len(unique_orders):
            sampled_orders = pd.Series(unique_orders).sample(
                n=sample_n_orders,
                random_state=42,
                replace=False,
            )
            df_orders = df_orders[df_orders[order_col].isin(sampled_orders)]
            print(
                f"Subsample: {sample_n_orders} orders "
                f"(from {len(unique_orders)} original)"
            )

    base_df_pop, basket = make_basket(
        df_orders=df_orders,
        order_col=order_col,
        product_col=product_col,
        min_item_freq=min_item_freq,
    )

    basket_bool = basket.astype(bool)
    n_transactions = basket_bool.shape[0]

    all_rules_ap = []
    all_rules_fp = []

    for s in min_supports:
        print(f"\n-- min_support = {s:.4f} --")

        freq_ap = apriori(
            basket_bool,
            min_support=s,
            use_colnames=True,
            max_len=max_itemset_len,
            low_memory=True,
        )
        if not freq_ap.empty:
            print(f"  Apriori itemsets: {len(freq_ap)}")
            rules_ap = association_rules(freq_ap, metric="lift", min_threshold=1.0)
            if not rules_ap.empty:
                rules_ap["algorithm"] = "apriori"
                rules_ap["support_level"] = s
                all_rules_ap.append(rules_ap)
        else:
            print("  Apriori: no itemsets at this support.")

        try:
            freq_fp = fpgrowth(
                basket_bool,
                min_support=s,
                use_colnames=True,
                max_len=max_itemset_len,
            )
            if not freq_fp.empty:
                print(f"  FP-Growth itemsets: {len(freq_fp)}")
                rules_fp = association_rules(
                    freq_fp,
                    metric="lift",
                    min_threshold=1.0,
                )
                if not rules_fp.empty:
                    rules_fp["algorithm"] = "fpgrowth"
                    rules_fp["support_level"] = s
                    all_rules_fp.append(rules_fp)
            else:
                print("  FP-Growth: no itemsets at this support.")
        except ValueError as e:
            print(f"  FP-Growth failed at support={s:.4f}: {e}")

    rules_ap_all = pd.concat(all_rules_ap, ignore_index=True) if all_rules_ap else pd.DataFrame()
    rules_fp_all = pd.concat(all_rules_fp, ignore_index=True) if all_rules_fp else pd.DataFrame()

    print(f"\nTotal Apriori rules: {len(rules_ap_all)}")
    print(f"Total FP-Growth rules: {len(rules_fp_all)}")

    rules_ap_all = enrich_rules(rules_ap_all, n_transactions)
    rules_fp_all = enrich_rules(rules_fp_all, n_transactions)

    pruned_ap = prune_rules(rules_ap_all, **prune_kwargs) if not rules_ap_all.empty else rules_ap_all
    pruned_fp = prune_rules(rules_fp_all, **prune_kwargs) if not rules_fp_all.empty else rules_fp_all

    if not pruned_ap.empty or not pruned_fp.empty:
        final_rules = pd.concat([pruned_ap, pruned_fp], ignore_index=True)
    else:
        final_rules = pd.DataFrame()

    if not final_rules.empty:
        final_rules["dataset"] = dataset_label

    print(f"\n  Final pruned rules: {final_rules.shape[0]} rows")

    return {
        "base_df_pop": base_df_pop,
        "basket": basket,
        "rules_ap_all": rules_ap_all,
        "rules_fp_all": rules_fp_all,
        "pruned_ap": pruned_ap,
        "pruned_fp": pruned_fp,
        "final_rules": final_rules,
    }


# =============================================================================
#  Pipeline entry points (what the instructor asked for)
# =============================================================================
def load_data(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data for a given dataset key: 'global' or 'ecommerce'.
    Uses kagglehub by default; can be modified to use local paths.

    Returns:
        df       → main working DataFrame
        df_copy  → copy reserved for association rules or extra analyses
    """
    if dataset_name not in DATA_CONFIG:
        raise ValueError(f"Unknown dataset_name={dataset_name}. Use 'global' or 'ecommerce'.")

    cfg = DATA_CONFIG[dataset_name]

    if kagglehub is None:
        raise RuntimeError(
            "kagglehub is not available. Either install it "
            "or modify load_data() to read from local CSV paths."
        )

    print(f"\n=== Loading {dataset_name} dataset from Kaggle ===")
    path = kagglehub.dataset_download(cfg["kaggle_id"])
    file_path = os.path.join(path, cfg["filename"])

    df = pd.read_csv(file_path, encoding=cfg["encoding"])
    df_copy = df.copy()

    print(f"{dataset_name} shape: {df.shape}")
    return df, df_copy


def preprocess_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Dataset-specific preprocessing / feature engineering (including target creation).
    """
    print(f"\n=== Preprocessing dataset: {dataset_name} ===")

    if dataset_name == "global":
        # Example: drop postal code as in original
        if "Postal Code" in df.columns:
            df = df.drop("Postal Code", axis=1)

        df = preprocess_global_add_high_value(df)

    elif dataset_name == "ecommerce":
        df = preprocess_ecommerce_reordered(df)

    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}.")

    print("Preprocessing complete.")
    return df


def train_models_for_dataset(df: pd.DataFrame, dataset_name: str):
    """
    Wrapper  = "train_model" step from the instructions.
    Calls the multi-model pipeline and returns trained models + metrics.
    """
    cfg = DATA_CONFIG[dataset_name]
    target_col = cfg["target_col"]
    drop_cols = cfg["drop_cols"]

    print(f"\n=== Training models for {dataset_name} ===")
    results_df, models_dict, prep = run_pipeline_for_all_models(
        df=df,
        target_col=target_col,
        drop_cols=drop_cols,
    )
    print("\nModel comparison table:")
    print(results_df.sort_values("test_roc_auc", ascending=False))

    return results_df, models_dict, prep


def run_association_rules_for_dataset(df_copy: pd.DataFrame, dataset_name: str):
    """
    Wrapper for association-rules pipeline per dataset.
    """
    print(f"\n=== Running association rules for {dataset_name} ===")

    if dataset_name == "global":
        # Basic cleaning similar to original script
        df_rules = df_copy.copy()
        if "Order Date" in df_rules.columns:
            df_rules["Order Date"] = pd.to_datetime(df_rules["Order Date"])
        key_cols = [c for c in ["Order ID", "Customer ID", "Order Date", "Sales", "Profit"] if c in df_rules.columns]
        df_rules = df_rules.dropna(subset=key_cols)
        df_rules = df_rules[(df_rules["Sales"] > 0) & (df_rules["Profit"] >= 0)]
        result = rule_based_pipeline(
            df_orders=df_rules,
            order_col="Order ID",
            product_col="Product Name",
            dataset_label="global_superstore",
            min_item_freq=3,
            min_supports=(0.0001, 0.0005),
            max_itemset_len=3,
            sample_n_orders=None,
        )

    elif dataset_name == "ecommerce":
        df_rules = df_copy.copy()
        df_rules = df_rules.dropna(subset=["order_id", "user_id", "product_id", "product_name"])
        df_rules["product_name"] = df_rules["product_name"].astype(str).str.strip().str.lower()
        result = rule_based_pipeline(
            df_orders=df_rules,
            order_col="order_id",
            product_col="product_name",
            dataset_label="ecommerce",
            min_item_freq=400,
            min_supports=(0.02,),
            max_itemset_len=2,
            sample_n_orders=None,
        )

    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}.")

    rules_df = result["final_rules"]
    print(f"\nSample rules for {dataset_name}:")
    print(rules_df.head())

    # Basic visualizations (lift bar / support-confidence scatter)
    if not rules_df.empty:
        top_k = min(10, len(rules_df))
        top_rules = rules_df.sort_values("lift", ascending=False).head(top_k)

        if "antecedent_str" in top_rules.columns:
            plt.figure(figsize=(12, 6))
            plt.barh(top_rules["antecedent_str"], top_rules["lift"])
            plt.xlabel("Lift")
            plt.ylabel("Antecedent")
            plt.title(f"Top {top_k} {dataset_name} Association Rules by Lift")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        if "support" in rules_df.columns and "confidence" in rules_df.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(rules_df["support"], rules_df["confidence"], alpha=0.3, s=10)
            plt.xlabel("Support")
            plt.ylabel("Confidence")
            plt.title(f"Support vs Confidence ({dataset_name})")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.show()

    return result


def run_pipeline_for_dataset(dataset_name: str):
    """
    Full pipeline for a single dataset:

        1) load_data
        2) preprocess_dataset
        3) optional EDA helpers (summary, distributions)
        4) train_models_for_dataset
        5) run_association_rules_for_dataset
    """
    print("=" * 80)
    print(f"STARTING FULL PIPELINE FOR DATASET: {dataset_name.upper()}")
    print("=" * 80)

    # 1) Load
    df, df_copy = load_data(dataset_name)

    # 2) Preprocess (includes target creation)
    df_pre = preprocess_dataset(df, dataset_name)

    # 3) Quick EDA (optional; can be commented out)
    print("\n--- EDA Summary ---")
    summary = data_preprocessing_summary(df_pre)
    print(summary.head())
    plot_numeric_distributions(df_pre, title=f"{dataset_name} numeric distributions")
    plot_correlation_heatmap(df_pre)

    # 4) Classification models
    results_df, models_dict, prep = train_models_for_dataset(df_pre, dataset_name)

    # 5) Association rules
    rules_result = run_association_rules_for_dataset(df_copy, dataset_name)

    return {
        "summary": summary,
        "results_df": results_df,
        "models": models_dict,
        "prep": prep,
        "rules_result": rules_result,
    }


def main():
    """
    Main entry point: run pipeline for both datasets.
    """
    # Global Superstore
    global_result = run_pipeline_for_dataset("global")

    # E-commerce Consumer Behaviour
    ecommerce_result = run_pipeline_for_dataset("ecommerce")

    print("\n=== Pipelines completed for both datasets. ===")


if __name__ == "__main__":
    main()
