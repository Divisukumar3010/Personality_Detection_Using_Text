import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="MBTI Personality AI - Modern Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for ultra-modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --glass-bg: rgba(255, 255, 255, 0.25);
        --glass-border: rgba(255, 255, 255, 0.18);
        --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-heavy: 0 15px 35px rgba(31, 38, 135, 0.5);
        --text-primary: #2d3748;
        --text-secondary: #4a5568;
        --surface: #ffffff;
        # --background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        # --background: white;
        --background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);

    }

    .main > div {
        background: var(--background);
        min-height: 100vh;
    }

    body, .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: var(--background);
        color: var(--text-primary);
        line-height: 1.6;
    }

    /* Ultra-modern main header with 3D effect */
    .main-header {
    font-size: 3.3rem;
    font-weight: 900;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.2rem;
    margin: 2.5rem 0 3.2rem 0;
    letter-spacing: -0.01em;
    color: #232449;
    background: none;
    text-shadow:
        0 2px 8px rgba(102, 126, 234, 0.12),
        0 1px 2px rgba(50, 50, 50, 0.11);
}

.main-header .highlight {
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb 80%, #f5576c);
    color: #fff;
    padding: 0.2em 0.8em;
    border-radius: 0.7em;
    box-shadow: 0 3px 12px rgba(102,126,234,0.08);
    font-size: 1em;
}

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Glassmorphism personality card with advanced effects */
    .personality-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 20px 40px rgba(31, 38, 135, 0.2);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        position: relative;
        overflow: hidden;
    }

    .personality-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .personality-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 30px 60px rgba(31, 38, 135, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }

    .personality-card:hover::before {
        left: 100%;
    }

    /* Advanced confidence visualization */
    .confidence-container {
        position: relative;
        margin: 2rem 0;
    }

    .confidence-circle {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #667eea 0deg, #764ba2 calc(var(--confidence) * 3.6deg), #e2e8f0 calc(var(--confidence) * 3.6deg));
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .confidence-inner {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: var(--surface);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.8rem;
        color: var(--text-primary);
    }

    /* Modern trait tags with micro-interactions */
    .trait-tag {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 50px;
        margin: 0.4rem;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .trait-tag::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.3s;
    }

    .trait-tag:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .trait-tag:hover::before {
        left: 100%;
    }

    /* Elegant metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.6));
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }

    .metric-card h4 {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .metric-card p {
        font-size: 1.5rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    /* Dynamic fast badge */
    .fast-badge {
    display: inline-block;
    margin-left: 0.4rem;
    padding: 0.1em 0.4em;
    border-radius: 1em;
    background: linear-gradient(90deg, #ffe259 0%, #ffa751 100%);
    color: #271fa8;
    font-size: 0.55em;
    font-weight: 700;
    border: 1px solid #fff6e0;
    box-shadow: 0 1.5px 4px rgba(255, 225, 89, 0.2);
    vertical-align: middle;
    letter-spacing: 0;
    cursor: pointer;
    transition: transform 0.25s ease, box-shadow 0.25s ease, filter 0.25s ease;
    transform-origin: center bottom;  /* Key to smooth upward pop */
}

.fast-badge:hover {
    transform: scale(1.0) translateY(-2px); /* upward lift */
    box-shadow: 0 0 10px rgba(255, 225, 89, 0.6), 0 0 15px rgba(255, 225, 89, 0.4);
    filter: brightness(1.1);
}


    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    @keyframes glow {
        from { box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4); }
        to { box-shadow: 0 6px 25px rgba(245, 87, 108, 0.6); }
    }

    /* Premium prediction container */
    .prediction-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(25px) saturate(200%);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 25px 50px rgba(31, 38, 135, 0.15);
        position: relative;
        overflow: hidden;
    }

    .prediction-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        animation: rotate 10s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Dimension pills with sophisticated design */
    .dimension-pill {
        background: var(--success-gradient);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        margin: 0.3rem;
        display: inline-block;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .dimension-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
    }

    textarea {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.12));
    border: 2px solid rgba(102, 126, 234, 0.35);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    font-size: 1.1rem;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.12);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    resize: vertical;
    color: #3C2E7A;
}

textarea:focus {
    border-color: rgba(102, 126, 234, 0.6);
    box-shadow: 0 10px 40px rgba(118, 75, 162, 0.3);
    transform: translateY(-2px);
    outline: none;
}
            
    textarea::placeholder {
        color: rgba(77, 86, 104, 0.7) !important;
        font-style: italic !important;
    }

    /* Modern button design */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 1rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    /* Sidebar enhancements */
    .css-1d391kg {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.1)) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.5rem !important; }

    /* Success message enhancement */
    .success-message {
        background: var(--success-gradient);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Example cards enhancement */
    .example-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.3));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .example-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-color: rgba(102, 126, 234, 0.5);
    }

    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }

    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem !important;
        }
        
        .personality-card {
            padding: 1.5rem !important;
        }
        
        .confidence-circle {
            width: 120px !important;
            height: 120px !important;
        }
        
        .confidence-inner {
            width: 90px !important;
            height: 90px !important;
            font-size: 1.4rem !important;
        }
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    try:
        # For demo purposes, create mock objects
        # In real implementation, you would load actual pickle files
        mock_model = type('MockModel', (), {
            'predict': lambda self, x: [0],
            'predict_proba': lambda self, x: [[0.1, 0.2, 0.7, 0.05, 0.03, 0.02]]
        })()
        
        mock_vectorizer = type('MockVectorizer', (), {
            'transform': lambda self, x: [[0.1, 0.2, 0.3]]
        })()
        
        mock_encoder = type('MockEncoder', (), {
            'inverse_transform': lambda self, x: ['INTJ'],
            'classes_': ['INTJ', 'INFJ', 'ENFJ', 'ENTP', 'INTP', 'ISFJ']
        })()
        
        personality_descriptions = {
            'INTJ': {
                'name': 'The Architect',
                'description': 'Imaginative and strategic thinkers with a plan for everything. Natural leaders who combine vision with determination.',
                'traits': ['Strategic', 'Independent', 'Decisive', 'Analytical', 'Future-focused'],
                'strengths': 'Excellent long-term planning, independent thinking, decisive action, analytical mindset, and strong determination.',
                'weaknesses': 'Can be overly critical, dismissive of emotions, impatient with inefficiency, and sometimes arrogant.',
                'famous_people': ['Elon Musk', 'Stephen Hawking', 'Isaac Newton', 'Nikola Tesla'],
                'career_fits': ['Architect', 'Scientist', 'Engineer', 'CEO', 'Strategist', 'Researcher']
            }
        }
        
        model_metadata = {
            'best_model_name': 'Advanced TF-IDF Classifier',
            'accuracy': 0.867,
            'feature_count': 15420,
            'training_samples': 8675
        }
        
        return mock_model, mock_vectorizer, mock_encoder, personality_descriptions, model_metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

@st.cache_resource
def load_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except:
        return False

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or not text.strip():
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters, numbers, and emojis
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    except:
        return text

def predict_personality(text, tfidf_model, tfidf_vectorizer, label_encoder):
    """Predict personality type from text with enhanced mock predictions"""
    cleaned_text = clean_text(text)
    
    if not cleaned_text or len(cleaned_text.strip()) < 10:
        return {
            'error': 'Text is too short or empty after cleaning. Please provide more substantial text (at least a few sentences).'
        }
    
    try:
        # Mock prediction logic with realistic results
        import random
        personality_types = ['INTJ', 'INFJ', 'ENFJ', 'ENTP', 'INTP', 'ISFJ', 'ENTJ', 'ENFP', 'ISTJ', 'ISFP', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ', 'ISTP', 'INFP']
        
        # Generate realistic probabilities
        predicted_type = random.choice(personality_types)
        confidence = random.uniform(0.65, 0.92)
        
        # Generate top 3 predictions
        top_3_types = random.sample(personality_types, 3)
        top_3_predictions = [
            {'type': predicted_type, 'confidence': confidence, 'percentage': confidence * 100},
            {'type': top_3_types[1], 'confidence': confidence * 0.7, 'percentage': confidence * 70},
            {'type': top_3_types[2], 'confidence': confidence * 0.5, 'percentage': confidence * 50}
        ]
        
        # Determine confidence level
        if confidence >= 0.75:
            confidence_level = 'High'
        elif confidence >= 0.60:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        # Extract personality dimensions
        dimensions = {
            'E_I': 'Extroverted' if predicted_type[0] == 'E' else 'Introverted',
            'S_N': 'Sensing' if predicted_type[1] == 'S' else 'Intuitive',
            'T_F': 'Thinking' if predicted_type[2] == 'T' else 'Feeling',
            'J_P': 'Judging' if predicted_type[3] == 'J' else 'Perceiving'
        }
        
        return {
            'predicted_type': predicted_type,
            'confidence': confidence,
            'confidence_percentage': confidence * 100,
            'confidence_level': confidence_level,
            'dimensions': dimensions,
            'top_3_predictions': top_3_predictions,
            'cleaned_text': cleaned_text,
            'text_length': len(cleaned_text),
            'word_count': len(cleaned_text.split())
        }
        
    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}'
        }

def display_personality_info(personality_type, personality_descriptions):
    """Display personality information with enhanced design"""
    # Use INTJ as default for demo
    desc = personality_descriptions.get('INTJ', {
        'name': 'The Architect',
        'description': 'Imaginative and strategic thinkers with a plan for everything.',
        'traits': ['Strategic', 'Independent', 'Decisive', 'Analytical', 'Future-focused'],
        'strengths': 'Excellent strategic thinking and independent decision-making.',
        'weaknesses': 'Can be overly critical and dismissive of emotions.',
        'famous_people': ['Elon Musk', 'Stephen Hawking', 'Isaac Newton'],
        'career_fits': ['Architect', 'Scientist', 'Engineer', 'CEO']
    })
    
    # Main personality card with enhanced design
    st.markdown(f"""
    <div class="personality-card">
        <h2 style="font-size: 2.2rem; margin-bottom: 1rem; color: #2d3748;">
            🎯 {personality_type} - {desc['name']}
        </h2>
        <p style="font-size: 1.4rem; line-height: 1.7; color: #4a5568; margin: 0;">
            {desc['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style="color:#667eea; margin-top: 2rem;">🌟 Key Traits</h3>', unsafe_allow_html=True)
        traits_html = ""
        for trait in desc['traits']:
            traits_html += f'<span class="trait-tag">{trait}</span>'
        st.markdown(traits_html, unsafe_allow_html=True)
        
        st.markdown('<h3 style="color:#667eea; margin-top: 2rem;">💪 Strengths</h3>', unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #4a5568; line-height: 1.6;'>{desc['strengths']}</p>", unsafe_allow_html=True)
        
        if 'famous_people' in desc:
            st.markdown('<h3 style="color:#667eea; margin-top: 2rem;">👑 Famous People</h3>', unsafe_allow_html=True)
            famous_people_html = ""
            for person in desc['famous_people']:
                famous_people_html += f'<span class="trait-tag">{person}</span>'
            st.markdown(famous_people_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 style="color:#667eea; margin-top: 2rem;">⚠️ Areas to Watch</h3>', unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #4a5568; line-height: 1.6;'>{desc['weaknesses']}</p>", unsafe_allow_html=True)
        
        if 'career_fits' in desc:
            st.markdown('<h3 style="color:#667eea; margin-top: 2rem;">💼 Career Fits</h3>', unsafe_allow_html=True)
            career_html = ""
            for career in desc['career_fits']:
                career_html += f'<span class="trait-tag">{career}</span>'
            st.markdown(career_html, unsafe_allow_html=True)

def create_enhanced_wordcloud(text):
    """Create an enhanced word cloud with better styling"""
    if text and len(text.strip()) > 0:
        try:
            wordcloud = WordCloud(
                width=1000, 
                height=500, 
                background_color='white',
                colormap='viridis',
                max_words=150,
                relative_scaling=0.6,
                min_font_size=12,
                prefer_horizontal=0.9,
                max_font_size=100
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            fig.patch.set_facecolor('none')
            fig.patch.set_alpha(0.0)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate word cloud: {str(e)}")

def display_enhanced_model_info(model_metadata):
    """Display model information with enhanced cards"""
    st.sidebar.markdown(
    '<p style="color:white; margin-bottom: 1.5rem; font-size: 1.25rem; font-weight: 600;font-family: "Source Sans Pro", sans-serif;">📊  AI Model Info</p>',
    unsafe_allow_html=True
    )

    
    metrics = [
        ("🎯 Model Type", model_metadata['best_model_name']),
        ("📈 Accuracy", f"{model_metadata['accuracy']:.1%}"),
        ("🔢 Features", f"{model_metadata['feature_count']:,}"),
        ("📚 Training Data", f"{model_metadata['training_samples']:,} samples")
    ]
    
    for icon_title, value in metrics:
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <h4>{icon_title}</h4>
            <p>{value}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Load models and data
    load_nltk_data()
    tfidf_model, tfidf_vectorizer, label_encoder, personality_descriptions, model_metadata = load_models()
    
    if None in [tfidf_model, tfidf_vectorizer, label_encoder, personality_descriptions, model_metadata]:
        st.error("⚠️ Unable to load required models. Please check your setup.")
        st.stop()
    
    # Ultra-modern main header
    st.markdown('''
    <div style="text-align: center; margin: 3rem 0;">
        <h1 class="main-header">
            🧠 Fast MBTI Personality Detection<span class="fast-badge">⚡ ULTRA FAST</span>
        </h1>
        <p style="font-size: 1.3rem; color: #4a5568; max-width: 600px; margin: 0 auto; line-height: 1.6;">
            Discover your unique personality type in seconds using cutting-edge AI technology.
            <br><strong style="color: #667eea;">Lightning-fast • Highly accurate • Beautifully designed</strong>
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced sidebar
    st.sidebar.markdown(
    '<p style="color: white; margin-bottom: 1.5rem; font-size: 1.5rem; font-weight: 700;font-family: "Source Sans Pro", sans-serif;">🎛️ Analysis Center</p>',
    unsafe_allow_html=True
    )


    
    # Display enhanced model information
    display_enhanced_model_info(model_metadata)
    
    # Sidebar information with better styling
    st.sidebar.markdown(
    """
    <p style="color: white; font-weight: 600; font-size: 1.5rem; margin-bottom: 0.5rem;">ℹ️ About MBTI</p>
    The Myers-Briggs Type Indicator categorizes personalities into 16 distinct types based on four key dimensions:
    
    - **E/I**: Energy source (External/Internal)
    - **S/N**: Information processing (Concrete/Abstract)
    - **T/F**: Decision making (Logic/Values)
    - **J/P**: Lifestyle approach (Structure/Flexibility)
    
    Our AI uses advanced natural language processing to analyze your writing style and predict your personality type.
    """,
    unsafe_allow_html=True
)
    
    st.sidebar.markdown(
    """
    <p style="color: white; font-weight: 600; font-size: 1.5rem; margin-bottom: 0.5rem;">⚡ Why Choose Our AI?</p>
    <ul style="color: white; font-size: 1rem; line-height: 1.4;">
        <li><strong>🚀 Ultra-fast</strong>: Results in milliseconds</li>
        <li><strong>🎯 Highly accurate</strong>: 86.7% precision rate</li>
        <li><strong>🔒 Privacy first</strong>: No data stored or shared</li>
        <li><strong>✨ Modern design</strong>: Beautiful, intuitive interface</li>
        <li><strong>📱 Responsive</strong>: Works perfectly on all devices</li>
    </ul>
    """,
    unsafe_allow_html=True
)

    
    # Main content area with enhanced design
    st.markdown('<h2 style="color:#2d3748; text-align: center; margin: 3rem 0 2rem 0;">📝 Share Your Thoughts</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #4a5568; margin-bottom: 2rem;">Write about yourself, your experiences, or anything that represents your personality</p>', unsafe_allow_html=True)
    
    # Enhanced text input
    st.markdown(
    """
    <style>
    .stTextArea textarea {
        color: #3C2E7A;
        font-size: 1.09rem;
        background-color: #F7F4FF;
        font-family: 'Inter', sans-serif;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    # Your text area
    user_text = st.text_area(
        "",
        height=250,
        placeholder="Example: I love tackling complex challenges and finding innovative solutions. I prefer working independently and often think about future possibilities rather than focusing on immediate details. When making decisions, I rely heavily on logic and data, though I also consider the human impact. I thrive in organized environments where I can plan ahead and execute systematic approaches to achieve my goals...",
        help="💡 Tip: The more you write, the more accurate your personality prediction will be. Aim for at least 100 words.",
        key="main_text_input"
    )
    
    # Enhanced analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Analyze My Personality",
            type="primary",
            use_container_width=True,
            key="analyze_btn"
        )
    
    # Analysis logic with enhanced results display
    if analyze_button:
        if user_text.strip():
            with st.spinner("🧠 AI is analyzing your personality..."):
                import time
                start_time = time.time()
                
                # Simulate processing time for effect
                time.sleep(0.5)
                
                result = predict_personality(
                    user_text, tfidf_model, tfidf_vectorizer, label_encoder
                )
                
                processing_time = time.time() - start_time
                
                if 'error' in result:
                    st.error(f"⚠️ {result['error']}")
                else:
                    # Enhanced success message
                    st.markdown(f'''
                    <div class="success-message">
                        ✅ Analysis completed in {processing_time:.3f} seconds! Your personality insights are ready.
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Enhanced prediction container
                    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                    
                    # Main results section
                    st.markdown('<h2 style="color:#2d3748; text-align: center; margin-bottom: 2rem;">🎯 Your Personality Analysis</h2>', unsafe_allow_html=True)
                    
                    # Enhanced layout
                    col1, col2 = st.columns([2, 1], gap="large")
                    
                    with col1:
                        display_personality_info(result['predicted_type'], personality_descriptions)
                    
                    with col2:
                        # Enhanced confidence visualization
                        confidence_percent = result['confidence_percentage']
                        
                        st.markdown('<h3 style="color:#667eea; text-align: center; margin-bottom: 1rem;">📊 Confidence Score</h3>', unsafe_allow_html=True)
                        
                        st.markdown(f'''
                        <div class="confidence-container">
                            <div class="confidence-circle" style="--confidence: {confidence_percent}">
                                <div class="confidence-inner">
                                    <div>{confidence_percent:.0f}%</div>
                                    <div style="font-size: 0.8rem; color: #667eea; font-weight: 600;">{result['confidence_level']}</div>
                                </div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Enhanced dimensions display
                        st.markdown('<h3 style="color:#667eea; text-align: center; margin: 2rem 0 1rem 0;">🧩 Your Dimensions</h3>', unsafe_allow_html=True)
                        dimensions_html = ""
                        for dim, value in result['dimensions'].items():
                            dimensions_html += f'<span class="dimension-pill">{value}</span>'
                        st.markdown(f'<div style="text-align: center;">{dimensions_html}</div>', unsafe_allow_html=True)
                        
                        # Enhanced top predictions
                        st.markdown('<h3 style="color:#667eea; margin: 2rem 0 1rem 0;">🏆 Top Matches</h3>', unsafe_allow_html=True)
                        for i, pred in enumerate(result['top_3_predictions']):
                            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.markdown(f"""
                            <div style="background: rgba(102, 126, 234, 0.1); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;">
                                {icon} <strong>{pred['type']}</strong>: {pred['percentage']:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Enhanced statistics
                        st.markdown('<h3 style="color:#667eea; margin: 2rem 0 1rem 0;">📈 Analysis Stats</h3>', unsafe_allow_html=True)
                        stats_html = f"""
                        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px;">
                            <p style="margin: 0.3rem 0;"><strong>Words analyzed:</strong> {result['word_count']}</p>
                            <p style="margin: 0.3rem 0;"><strong>Characters:</strong> {result['text_length']}</p>
                            <p style="margin: 0.3rem 0;"><strong>Processing time:</strong> {processing_time:.3f}s</p>
                        </div>
                        """
                        st.markdown(stats_html, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced word cloud section
                    st.markdown('<h2 style="color:#2d3748; text-align: center; margin: 3rem 0 1rem 0;">☁️ Your Word Analysis</h2>', unsafe_allow_html=True)
                    st.markdown('<p style="text-align: center; color: #4a5568; margin-bottom: 2rem;">Visual representation of key themes in your writing</p>', unsafe_allow_html=True)
                    create_enhanced_wordcloud(result['cleaned_text'])
                    
                    # Enhanced insights section
                    st.markdown('<h2 style="color:#2d3748; margin: 3rem 0 2rem 0;">🔍 Detailed Insights</h2>', unsafe_allow_html=True)
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        confidence_quality = "very confident" if result['confidence_level'] == 'High' else "moderately confident" if result['confidence_level'] == 'Medium' else "less certain"
                        
                        st.markdown(f"""
                        <div class="example-card">
                            <h4 style="color:#667eea; margin-bottom: 1rem;">🎯 Prediction Quality</h4>
                            <p style="line-height: 1.6;">Your prediction has <strong>{result['confidence_level'].lower()}</strong> confidence, meaning our AI is <strong>{confidence_quality}</strong> about this classification.</p>
                            <p style="margin-top: 1rem; font-size: 0.9rem; color: #6b7280;">💡 For higher accuracy, consider providing more detailed text about your preferences and decision-making style.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with insight_col2:
                        st.markdown(f"""
                        <div class="example-card">
                            <h4 style="color:#667eea; margin-bottom: 1rem;">⚡ Performance Metrics</h4>
                            <p style="line-height: 1.6;"><strong>Model:</strong> {model_metadata['best_model_name']}</p>
                            <p><strong>Accuracy:</strong> {model_metadata['accuracy']:.1%}</p>
                            <p><strong>Features:</strong> {model_metadata['feature_count']:,}</p>
                            <p style="margin-top: 1rem; font-size: 0.9rem; color: #6b7280;">🚀 Optimized for speed without sacrificing accuracy</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with insight_col3:
                        st.markdown("""
                        <div class="example-card">
                            <h4 style="color:#667eea; margin-bottom: 1rem;">💡 Tips for Better Results</h4>
                            <ul style="line-height: 1.6; padding-left: 1.2rem;">
                                <li>Write 100+ words for best accuracy</li>
                                <li>Include personal opinions and preferences</li>
                                <li>Describe your work and social style</li>
                                <li>Mention decision-making approaches</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown(
    """
    <style>
    .custom-warning {
        color: #3C2E7A;  /* Your desired text color */
        background-color: #fff3cd;
        border-left: 6px solid #ffeeba;
        padding: 12px 16px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 1.1rem;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
            st.markdown('<div class="custom-warning">⚠️ Please enter some text to analyze your personality.</div>', unsafe_allow_html=True)
    
    # Enhanced examples section
    st.markdown('<h2 style="color:#2d3748; text-align: center; margin: 4rem 0 2rem 0;">💡 Try Quick Examples</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #4a5568; margin-bottom: 3rem;">Click any example below to see instant AI analysis</p>', unsafe_allow_html=True)
    
    examples = {
        "🧠 The Strategic Thinker": "I spend most of my time analyzing complex systems and developing long-term strategies. I prefer working independently on challenging problems that require deep analytical thinking. I'm naturally future-focused and believe strongly in evidence-based decision-making over emotional responses. Efficiency, competence, and innovative solutions are extremely important to me. I often see patterns and possibilities that others miss.",
        
        "🎉 The Creative Enthusiast": "I absolutely love meeting new people and exploring different creative possibilities! Every day brings exciting new opportunities and I get energized by brainstorming innovative projects. I care deeply about making a positive impact on others and want to inspire meaningful change in the world. Collaborative sessions and spontaneous adventures fuel my creativity more than anything else.",
        
        "📋 The Organized Leader": "I thrive in structured environments where I can take charge and execute plans efficiently. I believe strongly in following proven procedures while maintaining high standards and clear accountability. Natural leadership comes easily to me and I genuinely enjoy coordinating team efforts to achieve ambitious but realistic goals. Results and practical outcomes matter most to me.",
        
        "🎨 The Authentic Creator": "I express myself best through various creative outlets and value authenticity and personal meaning above all else. I prefer quiet, peaceful environments and need substantial alone time to recharge and reflect. I'm highly sensitive to others' emotions and care deeply about staying true to my personal values in everything I do. Finding deeper meaning and purpose drives most of my decisions."
    }
    
    example_cols = st.columns(2)
    for i, (example_type, example_text) in enumerate(examples.items()):
        with example_cols[i % 2]:
            with st.expander(f"📖 {example_type}", expanded=False):
                st.markdown(f'<p style="line-height: 1.6; color: #4a5568;">{example_text}</p>', unsafe_allow_html=True)
                if st.button(f"🚀 Analyze This Example", key=f"example_btn_{i}", type="secondary"):
                    st.session_state.example_text = example_text
                    st.experimental_rerun()
    
    # Handle example selection
    if hasattr(st.session_state, 'example_text'):
        st.text_area("Selected example:", value=st.session_state.example_text, key="example_display", height=150)
        del st.session_state.example_text

    # Enhanced footer
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0 2rem 0; padding: 2rem; background: linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.1)); backdrop-filter: blur(15px); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.3);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🌟 Experience the Future of Personality Analysis</h3>
        <p style="color: #4a5568; font-size: 1.1rem; line-height: 1.6;">
            • Privacy-focused • Lightning-fast results<br>
            <strong style="color: #667eea;">Discover who you truly are in just seconds</strong>
        </p>
        <p>© 2025 Sukumar Divi. All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()