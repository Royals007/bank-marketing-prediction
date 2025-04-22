import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import data_preprocess


model_path = r"C:\Users\Roy\Documents\VisualCode\Python\bank-prediction-casestudy\models\model.pkl"
save_dir = r"C:\Users\Roy\Documents\VisualCode\Python\bank-prediction-casestudy\images"

def train_model(data_path, output_model_path=model_path):
    global y_test, y_pred, class_labels
    X, y, encoders, target_encoder = data_preprocess.preprocess_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_clf.fit(X_train, y_train)

    y_pred = model_clf.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_, digits=3))

   
    # Save report as csv file
    report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, digits=3, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, "classification_report.csv"), index=True)

    
    auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score: {auc:.2f}")


    joblib.dump({
        "model": model_clf,
        "encoders": encoders,
        "target_encoder": target_encoder
    }, output_model_path)

    print(f"Classification Model saved at: {output_model_path}")


# confusion matrix
def plot_confusion_matrix(y_test, y_pred, class_labels, save_dir=save_dir, file_name="confusion_matrix.png"):
    
    save_path = os.path.join(save_dir, file_name)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a heatmap for confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(save_path)
    print(f"Confusion matrix saved at: {save_path}")
    plt.show()

# Plot ROC curve
def plot_roc_curve(y_test, y_pred, save_dir=save_dir, file_name="roc_curve.png"):
    save_path = os.path.join(save_dir, file_name)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray') # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


train_model(data_path=data_preprocess.data_path, output_model_path=model_path)
class_labels = ['Yes', 'No'] # target variable is binary with labels 'Yes' and 'No'
plot_confusion_matrix(y_test, y_pred, class_labels)
plot_roc_curve(y_test, y_pred)
