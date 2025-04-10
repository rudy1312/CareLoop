from transformers import pipeline

# Load the summarization pipeline (only needs to be done once)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load the zero-shot classification pipeline (only needs to be done once)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
concern_categories = ["Staff", "Billing", "Food and Amenities", "Cleanliness", "Post Discharge Care", "Communication", "Efficiency", "Comfort and Privacy", "Digital Experience"]

# Load the sentiment analysis pipeline (only needs to be done once)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def summarize_text(text, max_length=150, min_length=30):
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def tag_concerns_top_n(text, categories=concern_categories, top_n=3):
    try:
        result = classifier(text, candidate_labels=categories)
        if result and result['labels'] and result['scores']:
            top_results = list(zip(result['labels'][:top_n], result['scores'][:top_n]))
            return top_results
        return None
    except Exception as e:
        print(f"Error tagging concern: {e}")
        return None

def analyze_sentiment_simplified(text):
    try:
        result = sentiment_analyzer(text)[0]
        star_label = result['label']
        score = result['score']

        if star_label in ['4 stars', '5 stars']:
            sentiment = 'positive'
        elif star_label in ['1 star', '2 stars']:
            sentiment = 'negative'
        else:  # '3 stars'
            sentiment = 'neutral'

        return sentiment, score
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return None, None