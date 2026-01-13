import os
from flask import Flask, request, jsonify
from crew import TravelCrew  # Ensure your crew.py has a class or function named TravelCrew

app = Flask(__name__)

@app.route('/plan', methods=['POST'])
def plan_trip():
    # Get user input from the JSON request
    data = request.get_json()
    origin = data.get('origin')
    destination = data.get('destination')
    interests = data.get('interests')

    # Trigger your CrewAI logic
    try:
        # Initialize your crew logic here
        crew_instance = TravelCrew() 
        result = crew_instance.run(origin, destination, interests)
        return jsonify({"itinerary": str(result)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Cloud Run provides a $PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)