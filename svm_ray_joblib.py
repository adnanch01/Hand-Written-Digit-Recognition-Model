from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np
from ray.util.joblib import register_ray
import joblib

# Load dataset
digits = load_digits()
param_space = {
    'C': np.logspace(-6, 6, 30),
    'gamma': np.logspace(-8, 8, 30),
    'tol': np.logspace(-4, -1, 30),
    'class_weight': [None, 'balanced'],
}

# Define model and search
model = SVC(kernel='rbf')
search = RandomizedSearchCV(model, param_space, cv=5, n_iter=300, verbose=10)

# Register Ray backend
register_ray()
with joblib.parallel_backend('ray'):
    search.fit(digits.data, digits.target)

print("Best parameters: ", search.best_params_)
