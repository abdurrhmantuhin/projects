import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# =============================================================================
# PART 1: GENERATE REALISTIC DATASET
# =============================================================================

def generate_dataset(n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Generate a realistic dataset of student performance metrics.
    
    Features: Study Hours, Sleep Hours, Quiz Scores, Attendance (%)
    Think of this as preprocessing raw data that might come from a CSV.
    """
    np.random.seed(seed)
    
    study_hours = np.random.normal(loc=5, scale=2, size=n_samples)
    study_hours = np.clip(study_hours, 1, 12)  # Between 1-12 hours
    
    sleep_hours = np.random.normal(loc=7, scale=1.5, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)  # Between 4-10 hours
    
    quiz_scores = 40 + 8 * study_hours + 5 * sleep_hours + \
                  np.random.normal(0, 5, n_samples)
    quiz_scores = np.clip(quiz_scores, 0, 100)
    
    attendance = 50 + 5 * study_hours + 3 * sleep_hours + \
                 np.random.normal(0, 8, n_samples)
    attendance = np.clip(attendance, 0, 100)
    
    # Stack all features: shape (n_samples, 4)
    data = np.column_stack((study_hours, sleep_hours, quiz_scores, attendance))
    return data


# =============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def explore_data(data: np.ndarray) -> dict:
    """
    Understand your data: mean, std, min, max, correlations.
    This builds intuition about what you're working with.
    """
    stats = {}
    
    # Basic statistics
    stats['mean'] = np.mean(data, axis=0)
    stats['std'] = np.std(data, axis=0)
    stats['min'] = np.min(data, axis=0)
    stats['max'] = np.max(data, axis=0)
    stats['median'] = np.median(data, axis=0)
    
    # Quantiles (useful for understanding distribution)
    stats['q25'] = np.percentile(data, 25, axis=0)
    stats['q75'] = np.percentile(data, 75, axis=0)
    
    # Identify outliers using IQR method
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
    stats['n_outliers'] = np.sum(outlier_mask)
    
    return stats, outlier_mask


def print_stats(stats: dict, feature_names: list):
    """Pretty print statistics"""
    print("=" * 70)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 70)
    for i, name in enumerate(feature_names):
        print(f"\n{name}:")
        print(f"  Mean: {stats['mean'][i]:.2f}")
        print(f"  Std:  {stats['std'][i]:.2f}")
        print(f"  Min:  {stats['min'][i]:.2f}")
        print(f"  Max:  {stats['max'][i]:.2f}")
        print(f"  Q25:  {stats['q25'][i]:.2f}")
        print(f"  Q75:  {stats['q75'][i]:.2f}")


# =============================================================================
# PART 3: DATA PREPROCESSING
# =============================================================================

def handle_outliers(data: np.ndarray, outlier_mask: np.ndarray, method: str = 'remove') -> np.ndarray:
    """
    Handle outliers: remove or clip them.
    This is a real preprocessing step you'll encounter in ML projects.
    """
    if method == 'remove':
        return data[~outlier_mask]
    elif method == 'clip':
        data_clipped = data.copy()
        for i in range(data.shape[1]):
            q1 = np.percentile(data[:, i], 25)
            q3 = np.percentile(data[:, i], 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            data_clipped[:, i] = np.clip(data[:, i], lower, upper)
        return data_clipped
    return data


# =============================================================================
# PART 4: FEATURE NORMALIZATION (CRUCIAL FOR ML)
# =============================================================================

def normalize_data(data: np.ndarray, method: str = 'zscore') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features so they're on the same scale.
    This is ESSENTIAL before most ML algorithms.
    
    Returns: normalized_data, mean, std (or min, max for minmax)
    """
    if method == 'zscore':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        normalized = (data - mean) / std
        return normalized, mean, std
    
    elif method == 'minmax':
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        normalized = (data - min_vals) / range_vals
        return normalized, min_vals, max_vals
    
    return data, None, None


# =============================================================================
# PART 5: FEATURE ENGINEERING
# =============================================================================

def engineer_features(data: np.ndarray) -> np.ndarray:
    """
    Create new features from existing ones.
    This teaches you how to think about data relationships.
    
    Original: Study Hours, Sleep Hours, Quiz Scores, Attendance
    New: Efficiency, Consistency Score, Potential
    """
    study = data[:, 0]
    sleep = data[:, 1]
    quiz = data[:, 2]
    attendance = data[:, 3]
    
    # Efficiency: Quiz score per study hour
    efficiency = quiz / (study + 1)  # +1 to avoid division by zero
    
    # Sleep efficiency: Quiz score per sleep hour
    sleep_efficiency = quiz / (sleep + 1)
    
    # Combined potential: weighted combination
    potential = 0.4 * quiz + 0.3 * attendance + 0.2 * efficiency + 0.1 * sleep_efficiency
    
    new_features = np.column_stack((efficiency, sleep_efficiency, potential))
    return new_features


# =============================================================================
# PART 6: CORRELATION ANALYSIS
# =============================================================================

def compute_correlation(data: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Understand relationships between features using correlation.
    This helps you understand which features matter for prediction.
    """
    # Normalize first (correlation requires same scale)
    normalized, _, _ = normalize_data(data)
    
    # Correlation matrix: dot product of normalized features
    correlation_matrix = np.dot(normalized.T, normalized) / len(data)
    
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            corr = correlation_matrix[i, j]
            print(f"{feature_names[i]} <-> {feature_names[j]}: {corr:.3f}")
    
    return correlation_matrix


# =============================================================================
# PART 7: DIMENSIONALITY REDUCTION (PCA-like concept)
# =============================================================================

def reduce_dimensionality(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce data to fewer dimensions while keeping variance.
    This is a simplified PCA concept to understand dimension reduction.
    """
    # Normal
    normalized, _, _ = normalize_data(data)
    
    # Compute covariance matrix
    cov_matrix = np.cov(normalized.T)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    top_eigenvectors = eigenvectors[:, :n_components]
    
    # Project data
    reduced_data = np.dot(normalized, top_eigenvectors)
    
    variance_explained = eigenvalues[:n_components] / np.sum(eigenvalues)
    print(f"\nVariance explained by {n_components} components: {variance_explained.sum():.2%}")
    
    return reduced_data


# =============================================================================
# PART 8: VISUALIZATION
# =============================================================================

def visualize_pipeline(original_data: np.ndarray, 
                       normalized_data: np.ndarray,
                       reduced_data: np.ndarray):
    """
    Visualize the transformation journey of your data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data: Study vs Quiz
    axes[0, 0].scatter(original_data[:, 0], original_data[:, 2], alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Study Hours')
    axes[0, 0].set_ylabel('Quiz Scores')
    axes[0, 0].set_title('Original Data: Study Hours vs Quiz Scores')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Normalized data
    axes[0, 1].scatter(normalized_data[:, 0], normalized_data[:, 2], alpha=0.6, s=30, color='orange')
    axes[0, 1].set_xlabel('Study Hours (normalized)')
    axes[0, 1].set_ylabel('Quiz Scores (normalized)')
    axes[0, 1].set_title('Normalized Data')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution of a feature
    axes[1, 0].hist(original_data[:, 2], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Quiz Scores')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Quiz Scores')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Reduced dimensions
    axes[1, 1].scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6, s=30, color='green')
    axes[1, 1].set_xlabel('Principal Component 1')
    axes[1, 1].set_ylabel('Principal Component 2')
    axes[1, 1].set_title('Reduced to 2D (PCA-like)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_pipeline.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'data_pipeline.png'")
    plt.show()


# =============================================================================
# PART 9: MAIN PIPELINE
# =============================================================================

def run_pipeline():
    """Execute the complete data analysis pipeline"""
    
    feature_names = ['Study Hours', 'Sleep Hours', 'Quiz Scores', 'Attendance (%)']
    
    print("\n" + "=" * 70)
    print("NUMPY DATA ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\n[1] Generating dataset...")
    data = generate_dataset(n_samples=1000)
    print(f"    Shape: {data.shape}")
    
    # Step 2: Explore
    print("\n[2] Exploring data...")
    stats, outlier_mask = explore_data(data)
    print_stats(stats, feature_names)
    print(f"    Outliers found: {stats['n_outliers']}")
    
    # Step 3: Handle outliers
    print("\n[3] Handling outliers...")
    data_clean = handle_outliers(data, outlier_mask, method='clip')
    print(f"    Cleaned data shape: {data_clean.shape}")
    
    # Step 4: Normalize
    print("\n[4] Normalizing data...")
    normalized, mean, std = normalize_data(data_clean, method='zscore')
    print(f"    Normalized mean: {np.mean(normalized, axis=0)}")
    print(f"    Normalized std:  {np.std(normalized, axis=0)}")
    
    # Step 5: Feature engineering
    print("\n[5] Engineering new features...")
    new_features = engineer_features(data_clean)
    combined_data = np.column_stack((data_clean, new_features))
    print(f"    New features shape: {new_features.shape}")
    print(f"    Combined data shape: {combined_data.shape}")
    
    # Step 6: Correlation
    print("\n[6] Computing correlations...")
    correlation = compute_correlation(data_clean, feature_names)
    
    # Step 7: Dimensionality reduction
    print("\n[7] Reducing dimensionality...")
    reduced = reduce_dimensionality(data_clean, n_components=2)
    
    # Step 8: Visualize
    print("\n[8] Creating visualizations...")
    visualize_pipeline(data_clean, normalized, reduced)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE! You now understand:")
    print("  • Data exploration and statistics")
    print("  • Outlier detection and handling")
    print("  • Feature normalization (CRUCIAL for ML)")
    print("  • Feature engineering")
    print("  • Correlation analysis")
    print("  • Dimensionality reduction")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()