
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
# For Global Superstore
df = pd.read_csv("Global_Superstore2.csv", encoding="latin1")

# For Consumer Behaviour 
# df = pd.read_csv("Superstore_consumer behaviour.csv", header=1, low_memory=False)


# Basic Exploration
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Null Counts ---")
print(df.isna().sum())

print("\n--- Summary (Numeric) ---")
print(df.describe())

print("\n--- Summary (Categorical) ---")
print(df.describe(include="O"))

# Missing Values Visualization
plt.figure(figsize=(10,6))
df.isna().sum().sort_values(ascending=False).plot(kind="bar")
plt.title("Missing Values per Column")
plt.show()

# Unique Values
for col in df.select_dtypes(include="object").columns[:10]:  # limit to first 10 for brevity
    print(f"{col}: {df[col].nunique()} unique values")

# Distributions of Numerical Features
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols].hist(bins=30, figsize=(15,10))
plt.suptitle("Numerical Distributions", fontsize=16)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 8. Categorical Exploration
if "Category" in df.columns:
    plt.figure(figsize=(8,5))
    df.groupby("Category")["Sales"].sum().sort_values().plot(kind="barh")
    plt.title("Sales by Category")
    plt.show()

if "department" in df.columns:
    plt.figure(figsize=(8,5))
    df["department"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Departments (Consumer Behaviour)")
    plt.show()

# Time-Based Analysis
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
    monthly_sales = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
    monthly_sales.plot(figsize=(12,5))
    plt.title("Monthly Sales Trend")
    plt.ylabel("Total Sales")
    plt.show()

# Customer Behavior (Consumer Behaviour dataset)
if "days_since_prior_order" in df.columns:
    plt.figure(figsize=(8,5))
    df["days_since_prior_order"].dropna().hist(bins=30)
    plt.title("Distribution of Days Since Prior Order")
    plt.xlabel("Days")
    plt.show()

if "reordered" in df.columns:
    reorder_rate = df["reordered"].mean()
    print(f"Reorder Rate: {reorder_rate:.2%}")
