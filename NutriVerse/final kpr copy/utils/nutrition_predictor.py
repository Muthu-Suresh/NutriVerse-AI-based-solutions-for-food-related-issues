import pickle
model = pickle.load(open('utils/nutrition_model.pkl', 'rb'))

def predict_nutrition(input_text):
    # Replace with your preprocessing
    return model.predict([input_text])[0]