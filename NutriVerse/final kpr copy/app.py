from flask import Flask, request, jsonify, render_template
from opencage.geocoder import OpenCageGeocode
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import folium
import joblib
import random
import os
import math
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

app = Flask(__name__)

# ------------- FOOD CLASSIFICATION SETUP -------------
df_classify = pd.read_csv("food_classification_dataset.csv")
df_classify["food"] = df_classify["food"].str.strip().str.lower()
STARS_FILE = "stars.txt"

def load_stars():
    if not os.path.exists(STARS_FILE):
        with open(STARS_FILE, "w") as f:
            f.write("0")
        return 0
    with open(STARS_FILE, "r") as f:
        return int(f.read().strip())

def save_stars(count):
    with open(STARS_FILE, "w") as f:
        f.write(str(count))

# ------------- ASHRAM FINDER SETUP -------------
key = '5d29c413161e4b2aaff259fb723499ae'
geocoder = OpenCageGeocode(key)
ashrams_df = pd.read_csv('data/ashrams.csv')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def geocode_address(address):
    result = geocoder.geocode(address, countrycode='IN')
    if result and result[0]['components'].get('country') == 'India':
        return result[0]['geometry']['lat'], result[0]['geometry']['lng']
    return None, None

def find_matching_ashrams(lat, lon, food_type, preferred_time, total_people):
    filtered = ashrams_df[
        (ashrams_df['food_types'].str.contains(food_type, case=False, na=False)) &
        (ashrams_df['preferred_times'].str.contains(preferred_time, case=False, na=False)) &
        (ashrams_df['max_capacity'] >= total_people)
    ].copy()
    if filtered.empty:
        return None, []
    filtered['distance'] = filtered.apply(
        lambda row: haversine(lat, lon, row['latitude'], row['longitude']), axis=1
    )
    sorted_df = filtered.sort_values('distance')
    return sorted_df.iloc[0].to_dict(), sorted_df.to_dict(orient='records')

def generate_map(lat, lon, ashrams):
    m = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker([lat, lon], tooltip="Your Location", icon=folium.Icon(color='red')).add_to(m)
    for a in ashrams:
        folium.Marker(
            [a['latitude'], a['longitude']],
            popup=a['name'],
            tooltip=f"{a['name']} ({a['distance']:.2f} km)"
        ).add_to(m)
    map_path = os.path.join("static", "ashram_map.html")
    m.save(map_path)
    return "/static/ashram_map.html"

# ------------- DIET & NUTRITION SETUP -------------
df_diet = pd.read_csv("regionally_valid_diet_meal_plan (1).csv")
input_cols = ['Age', 'Gender', 'BMI', 'Activity Level', 'Goal', 'Medical Conditions', 'Allergies', 'Dietary Preference', 'Region']
meal_cols = ['Breakfast', 'Lunch', 'Dinner', 'Snack 1', 'Snack 2']

le_dict = {}
for col in input_cols:
    if df_diet[col].dtype == 'object':
        le = LabelEncoder()
        df_diet[col] = le.fit_transform(df_diet[col].astype(str))
        le_dict[col] = le

df_diet['Meal Plan'] = df_diet[meal_cols].agg('|'.join, axis=1)
meal_encoder = LabelEncoder()
df_diet['Meal Plan Encoded'] = meal_encoder.fit_transform(df_diet['Meal Plan'])

Q = defaultdict(lambda: np.zeros(len(meal_encoder.classes_)))
states = df_diet[input_cols].astype(str).agg('-'.join, axis=1).values
actions = df_diet['Meal Plan Encoded'].values
for i in range(len(states)):
    Q[states[i]][actions[i]] = 1

df_nutrition = pd.read_csv("deficiency_recommendation_dataset.csv")

# ------------- SURPLUS FOOD PREDICTION SETUP -------------
model = joblib.load("model.pkl")

# ------------- ROUTES -------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/diet')
def diet_model():
    return render_template("diet_model.html")

@app.route('/nutrition')
def nutrition_model():
    return render_template("nutrition_model.html")

@app.route('/food_donation')
def food_donation():
    return render_template("food_donation.html")

@app.route('/food_classifier')
def food_classifier():
    return render_template('kids.html')

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    raw_items = data.get("items", "")
    items = [item.strip().lower() for item in raw_items.split(",")]

    feedback = []
    stars_change = 0
    valid_item = False

    for item in items:
        match = df_classify[df_classify["food"] == item]
        if not match.empty:
            valid_item = True
            category = match.iloc[0]["category"]
            if category == "healthy":
                stars_change += 1
                feedback.append(f"{item.capitalize()} is healthy! You got a golden star!")
            else:
                stars_change -= 1
                feedback.append(f"{item.capitalize()} is unhealthy! Minus one golden star.")
        else:
            feedback.append(f"I don't know about {item}. Try again!")

    current_stars = load_stars()
    current_stars += stars_change
    current_stars = max(current_stars, 0)
    save_stars(current_stars)

    return jsonify({"feedback": feedback, "stars_change": stars_change, "total_stars": current_stars})

@app.route("/get-stars", methods=["GET"])
def get_stars():
    return jsonify({"total_stars": load_stars()})

@app.route('/find_ashram', methods=['POST'])
def find_ashram():
    data = request.form
    address = data.get('address')
    total_people = int(data.get('total_people'))
    food_type = data.get('food_type')
    preferred_time = data.get('preferred_time')

    lat, lon = geocode_address(address)
    if lat is None or lon is None:
        return jsonify({'error': 'Could not locate the address in India.'}), 400

    nearest, matches = find_matching_ashrams(lat, lon, food_type, preferred_time, total_people)
    if not matches:
        return jsonify({'error': 'No matching ashram found.'}), 404

    map_url = generate_map(lat, lon, matches)
    return jsonify({'ashram': nearest, 'map_url': map_url})

@app.route('/diet_recommend', methods=['POST'])
def recommend_diet():
    try:
        user_data = request.get_json()
        user_input = {}
        for col in input_cols:
            value = user_data.get(col, "")
            if col in le_dict:
                known_values = list(le_dict[col].classes_)
                best_match, score = process.extractOne(value, known_values)
                if score >= 80:
                    value = le_dict[col].transform([best_match])[0]
                else:
                    return jsonify({'error': f"Invalid value for {col}. Try values like {known_values}"}), 400
            else:
                value = float(value)
            user_input[col] = value

        state_str = '-'.join([str(user_input[col]) for col in input_cols])
        best_action = int(np.argmax(Q[state_str])) if state_str in Q else random.randint(0, len(meal_encoder.classes_) - 1)

        meal_plan = meal_encoder.inverse_transform([best_action])[0]
        breakfast, lunch, dinner, snack1, snack2 = meal_plan.split('|')
        return jsonify({
            'Breakfast': breakfast,
            'Lunch': lunch,
            'Dinner': dinner,
            'Snack 1': snack1,
            'Snack 2': snack2
        })

    except Exception as e:
        return jsonify({'error': f"Error in processing data: {str(e)}"}), 500

@app.route('/nutrition_recommend', methods=['POST'])
def nutrition_recommend():
    try:
        user_data = request.get_json()
        deficiency = user_data.get("deficiency", "").strip()
        if not deficiency:
            return jsonify({'error': 'Deficiency input is missing or invalid.'}), 400
        recommendation = df_nutrition[df_nutrition['Deficiency'].str.contains(deficiency, case=False, na=False)]
        if recommendation.empty:
            return jsonify({'error': f'No recommendations found for {deficiency}'}), 400
        r = recommendation.iloc[0]
        return jsonify({
            'Deficiency': deficiency,
            'Recommended Food 1': r['Recommended Food 1'],
            'Recommended Food 2': r['Recommended Food 2'],
            'Note': r['Notes']
        })
    except Exception as e:
        return jsonify({'error': f"Error in processing data: {str(e)}"}), 500

@app.route('/surplus', methods=['GET', 'POST'])
def surplus_prediction():
    if request.method == 'POST':
        menu_type = request.form['menu_type']
        sales_count = int(request.form['sales_count'])
        time_of_day = request.form['time_of_day']
        day_of_week = request.form['day_of_week']
        is_holiday = 1 if request.form['is_holiday'] == 'Yes' else 0
        special_event = request.form['special_event']

        input_data = pd.DataFrame([{
            'menu_type': menu_type,
            'sales_count': sales_count,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'is_holiday': is_holiday,
            'special_event': special_event
        }])

        predicted_surplus = model.predict(input_data)[0]
        return render_template('surplus_food.html', prediction=f"{predicted_surplus:.2f}")
    return render_template('surplus_food.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)