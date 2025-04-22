# Bank Marketing Prediction Case Study

 Refer to the task.md file about the project information.

### How to Run

1. Set up Environment

code conda env create -f env.yml

2. Run model_pred.py file and this initiates
- Preprocess the data
- Train the model
- Evaluate performance

Save plots and reports in /images and /reports

3. Run Individual Modules

python src/data_preprocess.py
python src/model_training.py
python src/model_pred.py

## Model Performance

The model is evaluated using the following metrics:

- Accuracy

- Precision

- Recall

- F1 Score

- ROC AUC Score (with plot)

- Confusion Matrix (with plot)

Also, a complete report is exported to:

/images/classification_report.csv

## Visualizations

Performance visualizations (saved to /images/):

Confusion matrix

ROC curve

# References

Dataset: UCI Bank Marketing Data

Language: Python 3.12+

Libraries: pandas, scikit-learn, matplotlib, seaborn, Numpy, Jupyter Notebook

Task.md


