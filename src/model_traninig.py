import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import data_preprocess


model_path = r"C:\Users\Roy\Documents\VisualCode\Python\bank-prediction-casestudy\models\model.pkl"

def train_model(data_path, output_model_path=model_path):
    global y_test, y_pred, class_labels
    X, y, encoders, target_encoder = data_preprocess.preprocess_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_clf.fit(X_train, y_train)

    y_pred = model_clf.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_, digits=3))



    # Metrics calculation
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy:.3f}")
    recall = recall_score(y_test, y_pred)
    #print(f"Recall: {recall:.3f}")
    precision = precision_score(y_test, y_pred)
    #print(f"Precision: {precision:.3f}")
    

    joblib.dump({
        "model": model_clf,
        "encoders": encoders,
        "target_encoder": target_encoder
    }, output_model_path)

    print(f"Classification Model saved at: {output_model_path}")


# confusion matrix
save_path = r"C:\Users\Roy\Documents\VisualCode\Python\bank-prediction-casestudy\images\confusion_matrix.png"

def plot_confusion_matrix(y_test, y_pred, class_labels, save_path=save_path):
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


train_model(data_path=data_preprocess.data_path, output_model_path=model_path)
class_labels = ['Yes', 'No'] # target variable is binary with labels 'Yes' and 'No'
plot_confusion_matrix(y_test, y_pred, class_labels,save_path=save_path)
