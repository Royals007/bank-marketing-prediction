import joblib
import pandas as pd

trained_model_path = r"C:\Users\Roy\Documents\VisualCode\Python\bank-prediction-casestudy\models\model.pkl"

def predict(input_dict, model_path=trained_model_path):

    try:
        model_bundle = joblib.load(model_path)
        model = model_bundle['model']
        encoders = model_bundle['encoders']
        target_encoder = model_bundle['target_encoder']
        
        df = pd.DataFrame([input_dict])

        # Apply encoding label to categorical features
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        prediction = model.predict(df)
        return target_encoder.inverse_transform(prediction)[0]  #decode to original label (yes/no)
    

    except Exception as e:
        raise RuntimeError(f"Error loading model or encoders: {e}")
