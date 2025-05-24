import pickle
model = pickle.load(open('utils/diet_model.pkl', 'rb'))

def predict_diet(input_text):
    # Replace with your preprocessing
    return model.predict([input_text])[0]