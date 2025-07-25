"""
Data Cleaning and Preprocessing Module
Simple utilities for data preparation and cleaning
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency


def find_missing_data(df):
    """Find columns with missing values"""
    missing_counts = df.isna().sum()
    return missing_counts[missing_counts > 0].to_dict()


def separate_column_types(df, target_col=None, uniqueness_limit=20):
    """Separate columns into different types"""
    if target_col:
        df = df.drop(columns=[target_col])

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = [col for col in df.columns if col not in numeric_cols]

    # Separate numeric into continuous and discrete
    continuous_cols = []
    discrete_cols = []

    for col in numeric_cols:
        if df[col].nunique() >= uniqueness_limit or 'count' in col.lower():
            continuous_cols.append(col)
        else:
            discrete_cols.append(col)

    return continuous_cols, text_cols, discrete_cols


def fill_missing_values(df, numeric_strategy='median', text_strategy='most_frequent'):
    """Fill missing values using specified strategies"""
    df_clean = df.copy()

    continuous_cols, text_cols, discrete_cols = separate_column_types(df_clean)
    all_numeric = continuous_cols + discrete_cols

    # Fill numeric columns
    if all_numeric:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        df_clean[all_numeric] = numeric_imputer.fit_transform(df_clean[all_numeric])

    # Fill text columns
    if text_cols:
        text_imputer = SimpleImputer(strategy=text_strategy)
        df_clean[text_cols] = text_imputer.fit_transform(df_clean[text_cols])

    return df_clean


def find_outliers_iqr(df, columns, multiplier=1.5):
    """Detect outliers using IQR method"""
    stats = df[columns].describe()
    q1 = stats.loc['25%']
    q3 = stats.loc['75%']
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (df[columns] < lower_bound) | (df[columns] > upper_bound)
    return outlier_mask


def fix_outliers(df, columns, multiplier=1.5, method='median'):
    """Replace outliers with median or mean values"""
    df_fixed = df.copy()
    outlier_mask = find_outliers_iqr(df_fixed, columns, multiplier)

    for col in columns:
        if outlier_mask[col].any():
            if method == 'median':
                replacement = df_fixed[col].median()
            else:
                replacement = df_fixed[col].mean()

            df_fixed.loc[outlier_mask[col], col] = replacement

    return df_fixed


def calculate_numeric_correlations(df, columns):
    """Calculate correlation matrix for numeric columns"""
    if not columns:
        return pd.DataFrame()
    return df[columns].corr().abs()


def find_correlated_features(correlation_matrix, threshold=0.85):
    """Find highly correlated features to remove"""
    features_to_drop = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]

            if correlation_matrix.iloc[i, j] > threshold:
                features_to_drop.add(col2)

    return list(features_to_drop)


def calculate_categorical_independence(df, columns):
    """Calculate chi-square p-values for categorical features"""
    if len(columns) < 2:
        return pd.DataFrame()

    p_value_matrix = pd.DataFrame(
        np.ones((len(columns), len(columns))),
        index=columns,
        columns=columns
    )

    for i, col1 in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col2 = columns[j]

            contingency_table = pd.crosstab(df[col1], df[col2])
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                _, p_value, _, _ = chi2_contingency(contingency_table)
                p_value_matrix.iloc[i, j] = p_value
                p_value_matrix.iloc[j, i] = p_value

    return p_value_matrix


def find_dependent_categorical_features(p_value_matrix, significance_level=0.05):
    """Find categorical features that are dependent (low p-value)"""
    features_to_drop = set()

    for i in range(len(p_value_matrix.columns)):
        for j in range(i + 1, len(p_value_matrix.columns)):
            col1 = p_value_matrix.columns[i]
            col2 = p_value_matrix.columns[j]

            if p_value_matrix.iloc[i, j] < significance_level:
                features_to_drop.add(col2)

    return list(features_to_drop)


def standardize_features(df, columns):
    """Standardize numeric features"""
    df_scaled = df.copy()

    if columns:
        scaler = StandardScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        return df_scaled, scaler

    return df_scaled, None


def clean_dataset(train_df, test_df, target_col,
                  numeric_fill='median', text_fill='most_frequent',
                  outlier_factor=1.5, correlation_limit=0.85,
                  independence_level=0.05):
    """Complete data cleaning pipeline"""

    print(f"Starting data cleaning for target: {target_col}")
    print(f"Initial train shape: {train_df.shape}, test shape: {test_df.shape}")

    # Separate features and target
    X_train = train_df.drop(columns=[target_col]).copy()
    y_train = train_df[target_col].copy()
    X_test = test_df.drop(columns=[target_col]).copy()
    y_test = test_df[target_col].copy()

    # Get column types
    continuous_cols, text_cols, discrete_cols = separate_column_types(X_train)
    all_categorical = text_cols + discrete_cols

    print(f"Found {len(continuous_cols)} continuous, {len(all_categorical)} categorical features")
    print(f"Target distribution: {y_train.value_counts().to_dict()}")

    # Remove highly correlated numeric features
    if continuous_cols:
        corr_matrix = calculate_numeric_correlations(X_train, continuous_cols)
        correlated_features = find_correlated_features(corr_matrix, correlation_limit)

        # Limit removal to avoid removing too many features
        max_remove = max(1, len(continuous_cols) // 3)
        correlated_features = correlated_features[:max_remove]

        if correlated_features:
            print(f"Removing {len(correlated_features)} correlated numeric features: {correlated_features}")
            X_train = X_train.drop(columns=correlated_features)
            X_test = X_test.drop(columns=correlated_features)
            continuous_cols = [col for col in continuous_cols if col not in correlated_features]

    # Remove dependent categorical features
    if len(all_categorical) > 1:
        p_matrix = calculate_categorical_independence(X_train, all_categorical)
        dependent_features = find_dependent_categorical_features(p_matrix, independence_level)

        # Limit removal to avoid removing too many features
        max_remove = max(1, len(all_categorical) // 3)
        dependent_features = dependent_features[:max_remove]

        if dependent_features:
            print(f"Removing {len(dependent_features)} dependent categorical features: {dependent_features}")
            X_train = X_train.drop(columns=dependent_features)
            X_test = X_test.drop(columns=dependent_features)
            all_categorical = [col for col in all_categorical if col not in dependent_features]

    # Handle missing values
    missing_train = find_missing_data(X_train)
    missing_test = find_missing_data(X_test)

    if missing_train:
        print(f"Found missing values in {len(missing_train)} training columns")
    if missing_test:
        print(f"Found missing values in {len(missing_test)} test columns")

    X_train = fill_missing_values(X_train, numeric_fill, text_fill)
    X_test = fill_missing_values(X_test, numeric_fill, text_fill)

    # Handle outliers in continuous features
    if continuous_cols:
        print(f"Processing outliers in {len(continuous_cols)} continuous features")
        X_train = fix_outliers(X_train, continuous_cols, outlier_factor)
        X_test = fix_outliers(X_test, continuous_cols, outlier_factor)

    # Standardize continuous features
    if continuous_cols and len(continuous_cols) > 0:
        print(f"Standardizing {len(continuous_cols)} continuous features: {continuous_cols}")
        X_train, scaler = standardize_features(X_train, continuous_cols)
        if scaler is not None:
            X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
        else:
            X_test, _ = standardize_features(X_test, continuous_cols)

    clean_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    clean_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    print(f"Final train shape: {clean_train.shape}, test shape: {clean_test.shape}")
    print(f"Final target distribution: {clean_train[target_col].value_counts().to_dict()}")
    print("Data cleaning completed successfully")
    return clean_train, clean_test


def main():
    """Main execution function"""
    print("Processing News Popularity Dataset...")
    news_train = pd.read_csv('../data/news_popularity_train.csv')
    news_test = pd.read_csv('../data/news_popularity_test.csv')

    news_train_clean, news_test_clean = clean_dataset(
        news_train, news_test, 'popularity_category'
    )

    print("\nProcessing Heart Disease Dataset...")
    heart_train = pd.read_csv('../data/heart_1_train.csv')
    heart_test = pd.read_csv('../data/heart_1_test.csv')

    heart_train_clean, heart_test_clean = clean_dataset(
        heart_train, heart_test, 'chd_risk'
    )

    news_train_clean.to_csv('../data/news_popularity_train_processed.csv', index=False)
    news_test_clean.to_csv('../data/news_popularity_test_processed.csv', index=False)
    heart_train_clean.to_csv('../data/heart_1_train_processed.csv', index=False)
    heart_test_clean.to_csv('../data/heart_1_test_processed.csv', index=False)

    print("\nAll datasets cleaned and saved successfully!")


if __name__ == '__main__':
    main()