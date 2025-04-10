import os
from transcription import transcribe_audio
from ml_processing import summarize_text, tag_concerns_top_n, analyze_sentiment_simplified

AUDIO_FILE_PATH = "/Users/rudrarajkundu/Developer/CareLoop/ml/data/sample_audio.mp3"

def process_feedback(feedback_data):
    """
    Processes a single piece of patient feedback (either text or audio).

    Args:
        feedback_data (dict): A dictionary containing either 'text' or 'audio_path'.

    Returns:
        dict or None: A dictionary containing the processed feedback information, or None if an error occurs.
    """
    processed_info = {}

    if "audio_path" in feedback_data and feedback_data["audio_path"]:
        transcription_text = transcribe_audio(feedback_data["audio_path"])
        if transcription_text:
            processed_info["transcription"] = transcription_text
            summary = summarize_text(transcription_text)
            if summary:
                processed_info["summary"] = summary
                top_concerns = tag_concerns_top_n(summary)
                if top_concerns:
                    processed_info["concern_tags"] = [{"tag": tag, "score": score} for tag, score in top_concerns]
                    sentiment, sentiment_score = analyze_sentiment_simplified(summary)
                    if sentiment:
                        processed_info["sentiment"] = sentiment
                        processed_info["sentiment_score"] = sentiment_score
                        return processed_info
                    else:
                        print("Error analyzing sentiment for audio.")
                        return None
                else:
                    print("Error tagging concerns for audio.")
                    return None
            else:
                print("Error summarizing text from audio.")
                return None
        else:
            print("Error transcribing audio.")
            return None

    elif "text" in feedback_data and feedback_data["text"]:
        processed_info["transcription"] = "" # No transcription for direct text
        summary = summarize_text(feedback_data["text"])
        if summary:
            processed_info["summary"] = summary
            top_concerns = tag_concerns_top_n(summary)
            if top_concerns:
                processed_info["concern_tags"] = [{"tag": tag, "score": score} for tag, score in top_concerns]
                sentiment, sentiment_score = analyze_sentiment_simplified(summary)
                if sentiment:
                    processed_info["sentiment"] = sentiment
                    processed_info["sentiment_score"] = sentiment_score
                    return processed_info
                else:
                    print("Error analyzing sentiment for text.")
                    return None
            else:
                print("Error summarizing text.")
                return None

    else:
        print("No feedback text or audio path provided in the data.")
        return None

if __name__ == "__main__":
    # Example usage for audio feedback:
    audio_feedback = {"audio_path": AUDIO_FILE_PATH}
    processed_audio_feedback = process_feedback(audio_feedback)
    if processed_audio_feedback:
        print("\n--- Processed Audio Feedback ---")
        print(processed_audio_feedback)

    # Example usage for text feedback (uncomment to use):
    # text_feedback = {"text": "The nurses were very helpful and the room was clean. However, the food was quite bland."}
    # processed_text_feedback = process_feedback(text_feedback)
    # if processed_text_feedback:
    #     print("\n--- Processed Text Feedback ---")
    #     print(processed_text_feedback)