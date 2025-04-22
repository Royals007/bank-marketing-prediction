from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the data
data_path = r"C:\Users\Roy\Documents\VisualCode\Python\bank-prediction-casestudy\data\processed-data\bank-additional-full-processed.csv"

def preprocess_data(data_path):
    df = pd.read_csv(data_path, sep=";")
    print(df.head())
    print("Data loaded successfully.")
    print("preprocessed data shape:", df.shape)
    print("preprocessed data columns:", df.columns)
    print("preprocessed data types:", df.dtypes)
    print("preprocessed data info:")
    print(df.info())
    print("preprocessed data description:")
    print(df.describe())


    label_encoders = {}

    # Encode all categorical features except the target column 'y'
    for col in df.select_dtypes(include='object').columns:
        if col != 'y':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            label_encoders[col] = encoder

    # Encode the target column 'y'
    target_encoder = LabelEncoder()
    # yes → 1, no → 0
    df['y'] = target_encoder.fit_transform(df['y'])

    # Split features and target variable
    X = df.drop('y', axis=1)
    y = df['y']

    return X, y, label_encoders, target_encoder

#preprocess_data(data_path)
