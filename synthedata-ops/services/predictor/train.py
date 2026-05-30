import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def train_pipeline(data_path, model_out, pipeline_meta_out):
    print(f"📦 Loading preprocessed data from: {data_path}")
    df = pd.read_csv(data_path)

    # Target column check based on paper targets
    target_col = 'prognosis' if 'prognosis' in df.columns else 'Possible_Diseases'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )

    print("⚙️ Tuning hyperparameters via GridSearchCV...")
    param_grid = {
        'num_leaves': [5, 15, 31],
        'max_depth': [3, 5, -1],
        'learning_rate': [0.05, 0.1, 0.5],
        'n_estimators': [10, 50, 100]
    }

    grid = GridSearchCV(
        estimator=LGBMClassifier(verbose=-1),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f"🏆 Best Parameters Found: {grid.best_params_}")

    predictions = best_model.predict(X_test)
    print("\n📊 Classification Performance Matrix:")
    print(classification_report(y_test, predictions))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(best_model, model_out)

    meta = {
        "categorical_cols": categorical_cols,
        "feature_columns": feature_columns,
        "target_labels": np.unique(y).tolist()
    }
    joblib.dump(meta, pipeline_meta_out)
    print("💾 Model artifacts successfully exported.")

if __name__ == "__main__":
    train_pipeline(
        data_path="/data/synthetic/generated_patients.csv",
        model_out="artifacts/lightgbm_model.pkl",
        pipeline_meta_out="artifacts/pipeline_meta.pkl"
    )