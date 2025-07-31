import joblib

# Load model and features
def load_model_and_features():
    model = joblib.load("data/best_model.pkl")
    feature_order = joblib.load("data/feature_columns.pkl")
    return model, feature_order
