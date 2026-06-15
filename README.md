# Comment Category Prediction Challenge

## Project Overview

This project focuses on a comment category prediction task. The goal is to accurately classify comments into different categories. The project explores the dataset and applies various preprocessing steps, including filling missing values, regex-based text cleaning, and feature engineering to enhance the performance of the classification models.

Several machine learning models were evaluated for this task, ranging from simple baselines to complex ensembles:
*   Dummy Classifier
*   Logistic Regression
*   Naive Bayes (MultinomialNB)
*   LinearSVC
*   Voting Classifier Ensemble
*   Stacking Classifier Ensemble (with LightGBM as the meta-model)

The project handles an imbalanced dataset and uses techniques like class weighting and k-fold cross-validation with custom threshold tuning to improve the macro F1 score, particularly for rare labels.

## Dataset

The dataset used in this project consists of training, testing, and sample submission files. The data includes text comments and various other features, such as upvotes, downvotes, emoticons, and categorical demographic indicators (race, religion, gender, disability).

*   `train.csv`: Training data containing comments and their corresponding labels.
*   `test.csv`: Testing data to evaluate the final model.
*   `Sample.csv`: A sample submission format.

## Installation

To run this project, you need Python 3 installed. You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

The primary dependencies include:
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `lightgbm`

## Project Structure

The project has been refactored from a Jupyter Notebook into a standard Python project layout:

```
.
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── src/                    # Source code directory
│   ├── preprocessing.py    # Data cleaning and feature engineering logic
│   ├── train.py            # Model training and cross-validation pipeline
│   └── predict.py          # Logic for generating predictions on test data
```

## Usage

1.  **Data Preparation**: Ensure your dataset files (`train.csv`, `test.csv`, `Sample.csv`) are placed in the expected directory paths (or modify the paths in the scripts accordingly).
2.  **Training**: Run the training script to build the model pipeline, perform cross-validation, and tune the prediction thresholds.
    ```bash
    python src/train.py
    ```
3.  **Prediction**: Run the prediction script to load the trained model, process the test dataset, and generate the `submission.csv` file.
    ```bash
    python src/predict.py
    ```
