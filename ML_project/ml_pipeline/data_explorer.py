"""
Dataset Analysis and Visualization Module
Simple yet comprehensive data exploration utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns


def categorize_columns(df, target_col=None, threshold=20):
    """Separate columns into numeric and categorical types"""
    if target_col:
        df = df.drop(columns=[target_col])

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = [col for col in df.columns if col not in numeric_cols]

    # Split numeric into continuous and discrete
    continuous_cols = []
    discrete_cols = []

    for col in numeric_cols:
        if df[col].nunique() >= threshold or 'count' in col.lower():
            continuous_cols.append(col)
        else:
            discrete_cols.append(col)

    return continuous_cols, text_cols, discrete_cols


def get_numeric_stats(df, columns):
    """Generate statistical summary for numeric columns"""
    if not columns:
        return pd.DataFrame()
    return df[columns].describe().T


def get_categorical_stats(df, columns):
    """Generate summary for categorical columns"""
    if not columns:
        return pd.DataFrame()

    summary = []
    for col in columns:
        summary.append({
            'column': col,
            'non_null': df[col].count(),
            'unique_values': df[col].nunique(dropna=True)
        })
    return pd.DataFrame(summary).set_index('column')


def create_boxplot_grid(df, columns, cols_per_row=8, fig_width=18, fig_height=8):
    """Create grid of boxplots for numeric data"""
    if not columns:
        return

    n_cols = len(columns)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        df[col].plot.box(ax=axes[i])
        axes[i].grid(True, alpha=0.3, linestyle='--')

    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def create_histogram_grid(df, columns, cols_per_row=8, fig_width=18, fig_height=8):
    """Create grid of histograms for data distribution"""
    # Remove URL columns if present
    clean_columns = [col for col in columns if 'url' not in col.lower()]
    if not clean_columns:
        return

    n_cols = len(clean_columns)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, col in enumerate(clean_columns):
        df[col].dropna().hist(ax=axes[i], bins=10, rwidth=0.8, edgecolor='black')
        axes[i].set_title(col)
        axes[i].grid(False)

    # Hide empty subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_target_distribution(train_df, test_df, target_col, dataset_name="Dataset"):
    """Compare target distribution between train and test sets"""
    train_counts = train_df[target_col].value_counts().sort_index()
    test_counts = test_df[target_col].value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training data
    ax1.bar(train_counts.index.astype(str), train_counts.values, color='steelblue')
    ax1.set_title(f"{dataset_name} - Training Data")
    ax1.set_xlabel(target_col)
    ax1.set_ylabel("Count")
    ax1.tick_params(axis='x', rotation=45)

    # Test data
    ax2.bar(test_counts.index.astype(str), test_counts.values, color='orange')
    ax2.set_title(f"{dataset_name} - Test Data")
    ax2.set_xlabel(target_col)
    ax2.set_ylabel("Count")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def compute_correlation_matrix(df, columns):
    """Calculate correlation matrix for numeric columns"""
    if not columns:
        return pd.DataFrame()
    return df[columns].corr(method='pearson')


def plot_correlation_heatmap(corr_matrix, title="Correlation Matrix"):
    """Display correlation heatmap"""
    if corr_matrix.empty:
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compute_chi_square_matrix(df, columns):
    """Calculate chi-square test p-values for categorical columns"""
    if len(columns) < 2:
        return pd.DataFrame()

    n = len(columns)
    p_values = pd.DataFrame(np.ones((n, n)), index=columns, columns=columns)

    for i, col1 in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col2 = columns[j]
            cross_tab = pd.crosstab(df[col1], df[col2])
            if cross_tab.shape[0] > 1 and cross_tab.shape[1] > 1:
                chi2, p_val, dof, expected = chi2_contingency(cross_tab)
                p_values.iloc[i, j] = p_val
                p_values.iloc[j, i] = p_val

    return p_values


def plot_p_value_heatmap(p_matrix, title="Chi-Square Independence Test"):
    """Display p-value heatmap for categorical variables"""
    if p_matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(p_matrix.values, cmap='Reds', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(range(len(p_matrix.columns)))
    ax.set_yticks(range(len(p_matrix.index)))
    ax.set_xticklabels(p_matrix.columns, rotation=90)
    ax.set_yticklabels(p_matrix.index)

    # Add colorbar
    plt.colorbar(im)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def run_complete_analysis(train_data, test_data, target_column, analysis_name):
    """Execute comprehensive analysis on dataset"""
    print(f"--- {analysis_name} Analysis ---")

    # Categorize features
    numeric_features, categorical_features, ordinal_features = categorize_columns(train_data, target_column)
    all_categorical = categorical_features + ordinal_features

    # Numeric analysis
    if numeric_features:
        print("\nNumeric Features Summary:")
        numeric_summary = get_numeric_stats(train_data, numeric_features)
        print(numeric_summary)

        create_boxplot_grid(train_data, numeric_features)

        # Correlation analysis
        corr_matrix = compute_correlation_matrix(train_data, numeric_features)
        plot_correlation_heatmap(corr_matrix, f"{analysis_name} - Numeric Correlations")

    # Categorical analysis
    if all_categorical:
        print("\nCategorical Features Summary:")
        cat_summary = get_categorical_stats(train_data, all_categorical)
        print(cat_summary)

        create_histogram_grid(train_data, all_categorical)

        # Independence analysis
        p_value_matrix = compute_chi_square_matrix(train_data, all_categorical)
        plot_p_value_heatmap(p_value_matrix, "Categorical Independence Analysis")

    # Target distribution
    plot_target_distribution(train_data, test_data, target_column, analysis_name)


def main():
    """Main execution function"""
    # News dataset
    news_train = pd.read_csv('../data/news_popularity_train.csv')
    news_test = pd.read_csv('../data/news_popularity_test.csv')
    run_complete_analysis(news_train, news_test, 'popularity_category', 'News Popularity')

    print("\n" + "=" * 80 + "\n")

    # Heart disease dataset
    heart_train = pd.read_csv('../data/heart_1_train.csv')
    heart_test = pd.read_csv('../data/heart_1_test.csv')
    run_complete_analysis(heart_train, heart_test, 'chd_risk', 'Heart Disease Risk')


if __name__ == "__main__":
    main()