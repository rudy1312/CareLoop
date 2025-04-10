from flask import Flask, request, jsonify
from process_feedback import process_feedback
import os

app = Flask(__name__)

# Define the endpoint for processing feedback
@app.route('/process_feedback', methods=['POST'])
def handle_feedback():
    if request.is_json:
        # Handle JSON data (for text feedback)
        feedback_data = request.get_json()
        processed_result = process_feedback(feedback_data)
        return jsonify(processed_result)
    elif 'audio' in request.files:
        # Handle audio file upload
        audio_file = request.files['audio']
        # Save the audio file temporarily (you might want to use a unique name)
        audio_path = os.path.join('data', 'uploaded_audio.mp3') # Temporary save
        audio_file.save(audio_path)
        feedback_data = {"audio_path": audio_path}
        processed_result = process_feedback(feedback_data)
        # Clean up the temporary audio file
        os.remove(audio_path)
        return jsonify(processed_result)
    else:
        return jsonify({"error": "No feedback data (text or audio) provided."}), 400

if __name__ == '__main__':
    app.run(debug=True)