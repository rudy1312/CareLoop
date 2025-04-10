import whisper
import torch
import os

# Define the default audio file path
DEFAULT_AUDIO_PATH = "/Users/rudrarajkundu/Developer/CareLoop/ml/data/sample_audio.mp3"

# Tried Whisper on apple silicon but doesn't play nice with Metal Performance Shader yet
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Transcription device: {device}")

# Load the Whisper model (you can choose the size as needed: tiny, base, small, medium, large)
model = whisper.load_model("small", device=device)

def transcribe_audio(audio_path=DEFAULT_AUDIO_PATH):
    """
    Transcribes audio from the given file path using the Whisper model.
    Defaults to DEFAULT_AUDIO_PATH if no path is provided.

    Args:
        audio_path (str, optional): The path to the audio file.
                                     Defaults to DEFAULT_AUDIO_PATH.

    Returns:
        str or None: The transcribed text if successful, None otherwise.
    """
    if os.path.exists(audio_path):
        try:
            print(f"Transcribing audio file: {audio_path}")
            result = model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
    else:
        print(f"❌ Audio file not found at: {audio_path}")
        return None

if __name__ == "__main__":
    # Example usage (replace with your actual audio file path if needed)
    transcript = transcribe_audio()
    if transcript:
        print("\n--- Transcript ---")
        print(transcript)