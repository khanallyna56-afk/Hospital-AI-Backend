import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
import joblib
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

print("Training Clinical Risk Assessment Model")
print(f"Features: {config.model.CLINICAL_FEATURES}")
print(f"Test Size: {config.training.CLINICAL_TEST_SIZE}")
print(f"Random Forest Estimators: {config.training.RANDOM_FOREST_ESTIMATORS}")
print("-" * 50)


# -----------------------------
# 1. Load Real Dataset & Generate Synthetic Clinical Features
# -----------------------------
def generate_synthetic_clinical_data(n_samples=500, random_state=42):
    """
    Generate realistic clinical data using statistical distributions and
    correlations. Uses real breast cancer dataset as base and synthesizes
    clinical features using medical statistics and correlations.
    """
    np.random.seed(random_state)
    
    # Load real breast cancer dataset
    cancer_data = load_breast_cancer()
    print(f"Loaded real breast cancer dataset: {len(cancer_data.target)} samples")
    
    # Use real cancer labels (benign=0, malignant=1)
    n_samples = min(n_samples, len(cancer_data.target))
    cancer = cancer_data.target[:n_samples]
    
    # Generate AGE using realistic distribution
    # Normal distribution: mean=55, std=12 (typical cancer patient age)
    # Higher risk for older patients (correlation with cancer)
    base_age = np.random.normal(55, 12, n_samples)
    age_cancer_factor = cancer * np.random.normal(5, 3, n_samples)
    age = np.clip(base_age + age_cancer_factor, 25, 90).astype(int)
    
    # Generate GENDER (0=Female, 1=Male)
    # Breast cancer is more common in females, use real distribution
    gender = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    # Generate SMOKING with correlation to cancer risk
    # Base rate ~20%, higher in cancer cases
    smoking_prob = np.where(cancer == 1, 0.35, 0.15)
    smoking = np.random.binomial(1, smoking_prob)
    
    # Generate ALCOHOL with correlation to cancer risk
    # Base rate ~25%, slightly higher in cancer cases
    alcohol_prob = np.where(cancer == 1, 0.32, 0.20)
    alcohol = np.random.binomial(1, alcohol_prob)
    
    # Generate BMI using log-normal distribution
    # Mean BMI ~26, std=4.5
    # Higher BMI correlated with cancer risk
    base_bmi = np.random.lognormal(
        mean=np.log(26),
        sigma=0.18,
        size=n_samples
    )
    bmi_cancer_factor = cancer * np.random.normal(2, 1.5, n_samples)
    bmi = np.clip(base_bmi + bmi_cancer_factor, 15.0, 45.0)
    
    # Generate WBC (White Blood Cell count)
    # Normal range: 4-11 thousand/μL
    # Cancer patients often have elevated WBC
    base_wbc = np.random.gamma(shape=5, scale=1.4, size=n_samples)
    wbc_cancer_factor = cancer * np.random.normal(2, 1, n_samples)
    wbc = np.clip(base_wbc + wbc_cancer_factor, 3.0, 18.0)
    
    # Generate HEMOGLOBIN
    # Normal: Males 13.5-17.5, Females 12-15.5 g/dL
    # Cancer patients often have lower hemoglobin (anemia)
    base_hemoglobin = np.where(
        gender == 1,
        np.random.normal(15.5, 1.2, n_samples),  # Male
        np.random.normal(13.5, 1.0, n_samples)   # Female
    )
    hemoglobin_cancer_factor = cancer * np.random.normal(-1.5, 0.8, n_samples)
    hemoglobin = np.clip(base_hemoglobin + hemoglobin_cancer_factor, 8.0, 18.0)
    
    # Generate PLATELETS
    # Normal range: 150-400 thousand/μL
    # Can be elevated or decreased in cancer
    base_platelets = np.random.normal(250, 50, n_samples)
    platelet_cancer_factor = cancer * np.random.normal(40, 30, n_samples)
    platelets = np.clip(
        base_platelets + platelet_cancer_factor,
        100,
        500
    ).astype(int)
    
    # Add some noise and realistic variations
    # Age increases slightly for gender=1
    age = age + (gender * np.random.randint(0, 3, n_samples))
    
    # Smoking affects hemoglobin slightly
    hemoglobin = hemoglobin - (smoking * np.random.normal(0.3, 0.2, n_samples))
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'smoking': smoking,
        'alcohol': alcohol,
        'bmi': np.round(bmi, 1),
        'wbc': np.round(wbc, 1),
        'hemoglobin': np.round(hemoglobin, 1),
        'platelets': platelets,
        'cancer': cancer
    })
    
    return df


# Generate synthetic clinical dataset
print("\nGenerating synthetic clinical data based on real distributions...")
df = generate_synthetic_clinical_data(n_samples=500, random_state=42)

print(f"\nDataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['cancer'].value_counts())
print(f"\nDataset statistics:")
print(df.describe())

print(f"\nSample data (first 5 rows):")
print(df.head())

# -----------------------------
# 2. Split data
# -----------------------------
X = df.drop("cancer", axis=1)
y = df["cancer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config.training.CLINICAL_TEST_SIZE,
    random_state=config.training.CLINICAL_RANDOM_STATE,
    stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# -----------------------------
# 3. Train model
# -----------------------------
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=config.training.RANDOM_FOREST_ESTIMATORS,
    random_state=config.training.CLINICAL_RANDOM_STATE,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✓ Training complete")

# -----------------------------
# 4. Evaluate
# -----------------------------
print("\nEvaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# -----------------------------
# 5. Save model
# -----------------------------
config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, str(config.model.CLINICAL_MODEL_PATH))

print(f"\n✅ Clinical model saved at: {config.model.CLINICAL_MODEL_PATH}")
print("\n" + "="*50)
print("Training completed successfully!")
print("="*50)
