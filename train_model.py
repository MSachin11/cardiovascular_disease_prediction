import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib

# Load and preprocess data
data = pd.read_csv(r"E:\PROJECTS\cardiovascular_disease_prediction\cardio_train.csv", sep=';')
if 'id' in data.columns:
    data = data.drop(['id'], axis=1)

X = data[['age', 'height', 'weight', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
y = data['cardio']

# Handle imbalance using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with hyperparameter tuning
rf = RandomForestClassifier()
param_grid = {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2]}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(grid_search.best_estimator_, 'cardio_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
