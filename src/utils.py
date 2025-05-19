# Import necessary libraries
import pandas as pd                    # For data manipulation and analysis
import numpy as np                     # For numerical operations
import matplotlib.pyplot as plt        # For plotting
import seaborn as sns                  # For enhanced statistical visualizations
from scipy import stats                # For statistical computations like z-score
import os                              # For interacting with the file system

# Function to load CSV data from a specified path
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=",", encoding="utf-8")
        print(f"Successfully loaded: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

# Function to display summary statistics
def Summary_Statistics(df):
    print("\n--- Summary Statistics ---")
    print(df.describe())
# Function that Show count of missing values 
def missing_report(df):
    print("\n--- Missing Value Report ---")
    print(df.isna().sum())
# Function that Columns with >5% null
def null_value(df):                 
    null_percentage = (df.isna().sum() / len(df)) * 100
    high_null_cols = null_percentage[null_percentage > 5].index.tolist()
    print(f"\nColumns with >5% null values: {high_null_cols}") 
      

# Function to detect outliers using Z-score and flag them, and impute key columns
def outlier_detection(df):
    df['Cleaning Flag'] = 0                         # Initialize cleaning flag
    outlier_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']

    # Identify and flag outliers in specified columns
    for col in outlier_cols:
        if col in df.columns:
            df[f'{col}_ZScore'] = np.abs(stats.zscore(df[col]))  # Compute Z-score
            outlier_mask = df[f'{col}_ZScore'] > 3
            df.loc[outlier_mask, 'Cleaning Flag'] = 1
            print(f"Number of outliers in {col}: {outlier_mask.sum()}")
            df.drop(columns=[f'{col}_ZScore'], inplace=True)     # Clean up temporary column

    # Impute missing values in key solar metrics using median
    key_cols = ['GHI', 'DNI', 'DHI']
    for col in key_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Missing values in {col} imputed with median: {median_val}")

    print(f"\nTotal rows flagged for cleaning: {df['Cleaning Flag'].sum()}")

# Function to save the cleaned DataFrame to CSV
def export_cleaned_data(df, filename, output_dir='../data'):
    os.makedirs(output_dir, exist_ok=True)               # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data exported to: {output_path}")

# Function to perform time series analysis and visualizations
def time_series_analysis(df):
    if 'Timestamp' not in df.columns:
        print("Warning: 'Timestamp' column not found, skipping Time Series Analysis.")
        return
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])   # Convert to datetime
        df.set_index('Timestamp', inplace=True)             # Set datetime as index

        time_series_cols = ['GHI', 'DNI', 'DHI', 'Tamb']
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(time_series_cols, 1):
            if col in df.columns:
                plt.subplot(len(time_series_cols), 1, i)
                df[col].plot(title=f'{col} vs. Time')
                plt.ylabel(col)
        plt.tight_layout()
        plt.show()

        # Monthly distribution of GHI
        if 'GHI' in df.columns:
            df['Month'] = df.index.month
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Month', y='GHI', data=df)
            plt.title('GHI Distribution by Month')
            plt.show()
            df.drop(columns=['Month'], inplace=True)

            # Hourly GHI trend
            df['Hour'] = df.index.hour
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='Hour', y='GHI', data=df.groupby('Hour')['GHI'].mean().reset_index())
            plt.title('Average GHI Throughout the Day')
            plt.grid(True)
            plt.show()
            df.drop(columns=['Hour'], inplace=True)
    except KeyError as e:
        print(f"Error: Column '{e}' not found for Time Series Analysis.")
    except Exception as e:
        print(f"Unexpected error during Time Series Analysis: {e}")

# Compare average ModA and ModB before and after cleaning
def summary_statistics_missing_value(df):
    if 'Cleaning Flag' in df.columns and 'ModA' in df.columns and 'ModB' in df.columns:
        cleaning_impact = df.groupby('Cleaning Flag')[['ModA', 'ModB']].mean()
        print("\n--- Average ModA & ModB Pre/Post Cleaning ---")
        print(cleaning_impact)
        cleaning_impact.plot(kind='bar', title='Average ModA & ModB by Cleaning Flag')
        plt.ylabel('Average Value')
        plt.xticks(ticks=[0, 1], labels=['Original', 'Flagged'])
        plt.show()

# Function to perform and visualize correlation analysis
def correlation_analysis(df):
    corr_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB']
    valid_cols = [col for col in corr_cols if col in df.columns]
    if valid_cols:
        corr = df[valid_cols].corr()                         # Compute correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()
    # Scatter plot for Wind Speed vs GHI
    if 'WS' in df.columns and 'GHI' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='WS', y='GHI', data=df)
        plt.title('Wind Speed vs. GHI')
        plt.xlabel('Wind Speed (WS)')
        plt.ylabel('Global Horizontal Irradiance (GHI)')
        plt.show()

# Analyze wind direction and wind speed distribution
def wind_distribution_analysis(df):
    if 'WS' in df.columns and 'WD' in df.columns:
        try:
            wd_counts = df['WD'].value_counts().sort_index()
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, projection='polar')
            theta = np.linspace(0, 2 * np.pi, len(wd_counts), endpoint=False)
            width = 2 * np.pi / len(wd_counts)
            bars = ax.bar(theta, wd_counts.values, width=width, bottom=0.0)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(theta)
            ax.set_xticklabels(wd_counts.index)
            plt.title('Wind Direction Distribution (Radial Bar Plot)', va='bottom')
            plt.show()
        except Exception as e:
            print(f"An error occurred during Wind Direction analysis: {e}")

    # Histograms for GHI and Wind Speed
    hist_cols = ['GHI', 'WS']
    for col in hist_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

# Analyze relationship between humidity, temperature, and GHI
def temperature_analysis(df):
    if 'RH' in df.columns and 'Tamb' in df.columns and 'GHI' in df.columns:
        # Scatter plot: RH vs Tamb, colored by GHI
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df['RH'], df['Tamb'], c=df['GHI'], cmap='viridis')
        plt.title('Relative Humidity vs. Temperature (colored by GHI)')
        plt.xlabel('Relative Humidity (RH)')
        plt.ylabel('Ambient Temperature (Tamb)')
        plt.colorbar(sc, label='GHI')
        plt.show()

        # Scatter plot: RH vs GHI
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='RH', y='GHI', data=df)
        plt.title('Relative Humidity vs. GHI')
        plt.xlabel('Relative Humidity (RH)')
        plt.ylabel('Global Horizontal Irradiance (GHI)')
        plt.show()

# Bubble chart: GHI vs. Temperature with bubble size representing Humidity
def bubble_chart(df):
    if 'GHI' in df.columns and 'Tamb' in df.columns and 'RH' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Tamb', y='GHI', size='RH', alpha=0.6, data=df)
        plt.title('GHI vs. Temperature (Bubble size = RH)')
        plt.xlabel('Ambient Temperature (Tamb)')
        plt.ylabel('Global Horizontal Irradiance (GHI)')
        plt.show()
