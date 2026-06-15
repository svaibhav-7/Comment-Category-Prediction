import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import joblib
from preprocessing import enhance_features, preprocessor

def main():
    print("Loading data...")
    # These paths might need to be adjusted based on where the data actually resides
    train_df = pd.read_csv("/kaggle/input/datasets/sasivaibhav/dataset1/train.csv")

    print("Preprocessing data...")
    train_prepared = enhance_features(train_df)

    X = train_prepared.drop(columns=['label', 'created_date', 'comment'])
    y = train_prepared['label']

    print("Building model pipeline...")
    base_learners = [
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', C=4.0,random_state=42,n_jobs=-1,solver="saga")),
        ('svc', CalibratedClassifierCV(LinearSVC(class_weight='balanced', dual=False, C=0.1), cv=3))
    ]

    meta_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight={0:1,1:2,2:1.5,3:3})
    stack_ensemble = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model,
        passthrough=True,
        cv=5,
        n_jobs=-1
    )
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', stack_ensemble)
    ])

    print("Performing Stratified K-Fold to tune thresholds...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros((len(y), 4))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        final_pipeline.fit(X_tr, y_tr)

        oof_probs[val_idx] = final_pipeline.predict_proba(X_va)

    best_score = 0
    best_t1, best_t3 = 0, 0

    print("Tuning thresholds for label 1 and 3...")
    for t3 in np.arange(0.30,0.42,0.01):
        for t1 in np.arange(0.35, 0.48, 0.01):

            preds = []

            for p in oof_probs:
                if p[3] > t3:
                    preds.append(3)
                elif p[1] > t1:
                    preds.append(1)
                else:
                    preds.append(np.argmax(p))

            score = f1_score(y, preds, average="macro")

            if score > best_score:
                best_score = score
                best_t1, best_t3 = t1, t3

    print(f"Best Macro F1 Score: {best_score}")
    print(f"Best t1: {best_t1}, Best t3: {best_t3}")

    print("Training final model on all data...")
    final_pipeline.fit(X, y)

    print("Saving model and thresholds...")
    joblib.dump(final_pipeline, 'model.joblib')
    with open('thresholds.txt', 'w') as f:
        f.write(f"{best_t1},{best_t3}\n")

    print("Done!")

if __name__ == "__main__":
    main()
