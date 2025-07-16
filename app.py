
from flask import Flask, request, jsonify
import joblib
import math

app = Flask(__name__)

# Load your trained model
model = joblib.load("lgbm_final_model.joblib")

# Workout intensity info
INTENSITY_INFO = [
    {
        "level": "Very Low",
        "min": 0.0,
        "max": 2.0,
        "description": "Minimal exertion. Suitable for recovery or warm-up.",
        "examples": [
            "Gentle stretching",
            "Seated yoga or chair-based mobility",
            "Tai Chi",
            "Casual walking (under 3 km/h)",
            "Light household tasks (folding laundry, dusting)"
        ]
    },
    {
        "level": "Low",
        "min": 2.0,
        "max": 4.0,
        "description": "Easy, steady-state activity. Light elevation in heart rate.",
        "examples": [
            "Brisk walking (3–5 km/h)",
            "Beginner yoga or Pilates",
            "Easy cycling on flat ground",
            "Recreational swimming (slow pace)",
            "Light resistance band workouts"
        ]
    },
    {
        "level": "Moderate",
        "min": 4.0,
        "max": 6.0,
        "description": "Breathing heavier, can talk but not sing. Sustainable effort.",
        "examples": [
            "Jogging or slow running (6–8 km/h)",
            "Moderate cycling (outdoors or stationary)",
            "Zumba or dance cardio (low impact)",
            "Bodyweight circuits",
            "Resistance training with moderate weights",
            "Hiking on flat or slightly inclined terrain"
        ]
    },
    {
        "level": "High",
        "min": 6.0,
        "max": 8.0,
        "description": "Challenging. Short bursts or sustained hard effort.",
        "examples": [
            "Running at moderate speed (8–10 km/h)",
            "Swimming laps continuously",
            "HIIT sessions with short recovery",
            "Circuit training with minimal rest",
            "CrossFit-style moderate sessions",
            "Heavy resistance training (supersets)"
        ]
    },
    {
        "level": "Very High",
        "min": 8.0,
        "max": 10.0,
        "description": "Near max effort. Anaerobic, high heart rate, unsustainable for long.",
        "examples": [
            "Sprint intervals (e.g., 30s sprint / 30s rest)",
            "Tabata (e.g., 20s max effort, 10s rest)",
            "Advanced HIIT (burpees, jump squats, mountain climbers)",
            "Competitive sports (football, boxing, tennis matches)",
            "Stair sprints or hill climbs",
            "Powerlifting or Olympic lifting sets"
        ]
    }
]

# Calculate scaled Intensity Index
def calculate_scaled_intensity_index(heart_rate, duration, weight):
    min_ii = 0.494975
    max_ii = 196.420577

    ii = (heart_rate * duration * math.sqrt(weight)) / 1000
    scaled_ii = 1 + 9 * ((ii - min_ii) / (max_ii - min_ii))
    return ii, scaled_ii

# Get Intensity Level details
def get_intensity_info(scaled_ii):
    for item in INTENSITY_INFO:
        if item["min"] <= scaled_ii < item["max"]:
            return item
    return INTENSITY_INFO[-1]

@app.route('/')
def index():
    return 'Calorie Predictor API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        age = int(data['Age'])
        gender = int(data['Gender'])
        height = float(data['Height'])
        weight = float(data['Weight'])
        bmi = float(data['BMI'])
        duration = float(data['Duration'])
        heart_rate = float(data['Heart_Rate'])
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'Invalid or missing input. Please check your JSON payload.'}), 400

    ii, scaled_ii = calculate_scaled_intensity_index(heart_rate, duration, weight)
    intensity_info = get_intensity_info(scaled_ii)

    input_features = [[
        age, gender, height, weight, bmi, duration, scaled_ii, heart_rate
    ]]

    prediction = model.predict(input_features)[0]

    return jsonify({
        'predicted_calories': prediction,
        'scaled_intensity_index': scaled_ii,
        'intensity_level': intensity_info["level"],
        'intensity_description': intensity_info["description"],
        'intensity_examples': intensity_info["examples"]
    })

if __name__ == '__main__':
    app.run(debug=True)
