import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.predict import predict_sentiment

st.set_page_config(
    page_title="Text Sentiment Analysis", 
    layout="centered",
    page_icon="✏️",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #1DA1F2;
    }
    .stButton button {
        background: linear-gradient(90deg, #1DA1F2 0%, #0d8bd9 100%);
        color: white;
        font-weight: bold;
        border-radius: 25px;
        padding: 10px 30px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Header with animated styling
st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); margin-bottom: 30px;'>
        <h1 style='text-align: center; color: #1DA1F2; margin: 0;'>
            ✏️ Text Sentiment Analysis
        </h1>
        <p style='text-align: center; color: #657786; margin-top: 10px; font-size: 16px;'>
            Powered by AI • Analyze emotions in text instantly
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("""
    This application uses:
    - **Logistic Regression** model
    - **TF-IDF Vectorization**
    - Trained on 1.6M tweets
    """)
    
    st.markdown("### 📊 Model Stats")
    st.metric("Training Samples", "1.6M")
    st.metric("Features", "TF-IDF")
    st.metric("Algorithm", "Logistic Reg.")

# Main input section with card styling
st.markdown("""
    <div style='background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
        <h3 style='color: #1DA1F2; margin-top: 0;'>📝 Enter Your Text</h3>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

tweet = st.text_area(
    "Type or paste text below:", 
    height=150,
    placeholder="e.g., I absolutely love this product! It exceeded all my expectations. 😊",
    label_visibility="collapsed",
    max_chars=500
)

# Character counter
char_count = len(tweet)
st.caption(f"Characters: {char_count}/500")

# Center the button with better spacing
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")

if analyze_button:
    if tweet.strip() == "":
        st.warning("⚠️ Please enter some text to analyze")
    else:
        with st.spinner("🔄 Analyzing sentiment..."):
            result = predict_sentiment(tweet)

        # Results card
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                <h3 style='color: #1DA1F2; margin-top: 0;'>📊 Analysis Results</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            if result["sentiment"] == "positive":
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; text-align: center;'>
                        <p style='color: white; margin: 0; font-size: 14px;'>Sentiment</p>
                        <h1 style='color: #4ade80; margin: 10px 0; font-size: 48px;'>😊</h1>
                        <h2 style='color: white; margin: 0;'>Positive</h2>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; text-align: center;'>
                        <p style='color: white; margin: 0; font-size: 14px;'>Sentiment</p>
                        <h1 style='color: #f87171; margin: 10px 0; font-size: 48px;'>😞</h1>
                        <h2 style='color: white; margin: 0;'>Negative</h2>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            confidence_pct = result["confidence"] * 100
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; text-align: center;'>
                    <p style='color: white; margin: 0; font-size: 14px;'>Confidence Score</p>
                    <h1 style='color: #fbbf24; margin: 20px 0; font-size: 56px;'>{confidence_pct:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Enhanced progress bar
        st.markdown("**Confidence Level:**")
        st.progress(result["confidence"])
        
        # Interpretation
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        if result["confidence"] > 0.8:
            confidence_level = "Very High"
            emoji = "🎯"
        elif result["confidence"] > 0.6:
            confidence_level = "High"
            emoji = "✅"
        elif result["confidence"] > 0.5:
            confidence_level = "Moderate"
            emoji = "⚖️"
        else:
            confidence_level = "Low"
            emoji = "⚠️"
        
        st.info(f"{emoji} **Confidence Level:** {confidence_level} - The model is {'very certain' if result['confidence'] > 0.8 else 'reasonably confident' if result['confidence'] > 0.6 else 'somewhat uncertain'} about this prediction.")

# Examples section
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
with st.expander("💡 See Example Texts"):
    st.markdown("""
    **Positive Examples:**
    - "This is absolutely amazing! I couldn't be happier with the results."
    - "Wonderful experience, highly recommend to everyone!"
    
    **Negative Examples:**
    - "This is terrible and completely disappointing."
    - "I'm very frustrated with this poor quality product."
    """)

# Footer
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 20px; background: white; border-radius: 15px;'>
        <p style='color: #657786; font-size: 12px; margin: 0;'>
            ⚡ Powered by Logistic Regression & TF-IDF Vectorization<br>
            Made with ❤️ using Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)
