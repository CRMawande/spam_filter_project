import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# CONFIG
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# LOAD TRAINING DATA
X_train_count = joblib.load(os.path.join(MODELS_DIR, "X_train_count.pkl"))
y_train = joblib.load(os.path.join(MODELS_DIR, "y_train.pkl"))

# MULTINOMIALNB MODEL
mnb = MultinomialNB()
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5], 'fit_prior': [True, False]}

# GRID SEARCH
grid = GridSearchCV(mnb, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train_count, y_train)

best_mnb = grid.best_estimator_
print(f"[TRAINING] Best Params for MNB: {grid.best_params_}")

# SAVE BEST MODEL
joblib.dump(best_mnb, os.path.join(MODELS_DIR, "MNB_best.pkl"))
print("[TRAINING] Saved best model: MNB_best.pkl")
