# Model configurations
MODEL_CONFIGS = {
    'distilroberta': {
        'name': 'DistilRoBERTa',
        'model_id': 'j-hartmann/emotion-english-distilroberta-base',
        'type': 'emotion'
    },
    'vader': {
        'name': 'VADER',
        'type': 'sentiment'
    }
}

# Emotion labels
EMOTION_LABELS = {
    'joy': 'Joy',
    'sadness': 'Sadness',
    'anger': 'Anger',
    'fear': 'Fear',
    'surprise': 'Surprise',
    'neutral': 'Neutral',
    'love': 'Love',
    'disgust': 'Disgust',
    'optimism': 'Optimism',
    'pessimism': 'Pessimism',
    'trust': 'Trust',
    'anticipation': 'Anticipation'
}

# Application settings
APP_TITLE = "Emotion Analysis"
APP_DESCRIPTION = """
This application analyzes emotions in text using natural language processing models.
You can enter text directly or upload a file for analysis.
The analysis is performed using DistilRoBERTa for detailed emotion detection and VADER for sentiment analysis.
"""

# Visualization settings
CHART_HEIGHT = 500
CHART_WIDTH = 800

# Input settings
TEXT_AREA_HEIGHT = 150
MAX_FILE_SIZE = 1024 * 1024  # 1MB
ALLOWED_FILE_TYPES = ['txt']

# UI settings
SIDEBAR_HEADER = "Input Options"
INPUT_METHOD_LABEL = "Choose input method:"
INPUT_METHODS = ["Text Input", "File Upload"]
TEXT_INPUT_LABEL = "Enter your text:"
FILE_UPLOAD_LABEL = "Upload a text file"
FILE_CONTENT_LABEL = "File content:"
ANALYZE_TEXT_BUTTON = "Analyze Text"
ANALYZE_FILE_BUTTON = "Analyze File"

# Results section labels
RESULTS_HEADER = "Analysis Results"
DETAILED_SCORES_HEADER = "Detailed Scores"
MODEL_COMPARISON_HEADER = "Model Comparison"
SUMMARY_TABLE_HEADER = "Summary Table"
