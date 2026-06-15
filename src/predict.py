import pandas as pd
import numpy as np
import joblib
from preprocessing import enhance_features

def main():
    print("Loading test data...")
    test_df = pd.read_csv("/kaggle/input/datasets/sasivaibhav/dataset1/test.csv")
    sample_df = pd.read_csv("/kaggle/input/datasets/sasivaibhav/dataset1/Sample.csv")

    print("Preprocessing test data...")
    test_prepared = enhance_features(test_df)
    test_X = test_prepared.drop(columns=['created_date', 'comment'], errors='ignore')

    print("Loading model and thresholds...")
    final_pipeline = joblib.load('model.joblib')

    with open('thresholds.txt', 'r') as f:
        line = f.read().strip()
        best_t1, best_t3 = map(float, line.split(','))

    print("Generating predictions...")
    test_probs = final_pipeline.predict_proba(test_X)

    final_preds = []
    for p in test_probs:
        if p[3] > best_t3:
            final_preds.append(3)
        elif p[1] > best_t1:
            final_preds.append(1)
        else:
            final_preds.append(np.argmax(p))

    print("Saving submission.csv...")
    submission = pd.DataFrame({
        'ID': sample_df['ID'],
        'label': final_preds
    })
    submission.to_csv("submission.csv", index=False)

    print("Done")

if __name__ == "__main__":
    main()
