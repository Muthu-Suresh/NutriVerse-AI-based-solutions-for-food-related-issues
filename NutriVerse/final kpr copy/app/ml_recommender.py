import joblib

def predict_priority(distance, capacity, veg_match, time_match):
    model = joblib.load('ml_model/model.pkl')
    prediction = model.predict([[distance, capacity, veg_match, time_match]])
    return prediction[0]
