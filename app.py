import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from PIL import Image
import easyocr
import cv2
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Amazon Price Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .feature-item {
        background-color: #e6f3ff;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        border-left: 4px solid #FF9900;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛍️ Amazon Smart Price Predictor</div>', unsafe_allow_html=True)


# Initialize EasyOCR reader (cached for performance)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])


# OCR function with EasyOCR
def extract_text_from_image(image_file):
    """Extract text from product image using EasyOCR"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name

        # Load EasyOCR reader
        reader = load_reader()

        # Read text from image
        results = reader.readtext(tmp_path)

        # Combine all detected text
        extracted_text = ' '.join([result[1] for result in results])

        # Clean up
        os.unlink(tmp_path)

        return extracted_text

    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""


# Feature extraction function (same as before)
def extract_features_from_ocr_text(ocr_text):
    """Extract pricing-relevant features from OCR text"""
    features = {
        'product_name': '',
        'weight_value': None,
        'weight_unit': '',
        'num_bullet_points': 0,
        'description_length': len(ocr_text),
        'is_organic': 0,
        'is_gluten_free': 0,
        'is_vegan': 0,
        'is_kosher': 0,
        'is_non_gmo': 0,
        'has_premium_words': 0,
        'has_health_claims': 0,
        'has_origin_info': 0,
        'has_pack_info': 0
    }

    if not ocr_text.strip():
        return features

    text_lower = ocr_text.lower()

    # Extract product name (first few words often contain name)
    words = ocr_text.split()[:10]  # First 10 words
    features['product_name'] = ' '.join(words)

    # Extract weight information using regex patterns
    weight_patterns = [
        r'(\d+\.?\d*)\s*(oz|ounce|lb|pound|g|gram|kg|kilogram)',
        r'(\d+\.?\d*)\s*(fl\s*oz|fluid\s*ounce)',
        r'net\s*wt?\.?\s*(\d+\.?\d*)\s*(oz|ounce|g|gram)',
        r'(\d+)\s*(oz|ounce|lb|pound)'
    ]

    for pattern in weight_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                features['weight_value'] = float(matches[0][0])
                features['weight_unit'] = matches[0][1]
                break
            except ValueError:
                continue

    # Count bullet points and other indicators
    bullet_count = len(re.findall(r'[•\-*>\s]\s*.+', ocr_text))
    features['num_bullet_points'] = min(bullet_count, 10)  # Cap at 10

    # Health & dietary features
    features['is_organic'] = 1 if 'organic' in text_lower else 0
    features['is_gluten_free'] = 1 if any(phrase in text_lower for phrase in ['gluten-free', 'gluten free']) else 0
    features['is_vegan'] = 1 if 'vegan' in text_lower else 0
    features['is_kosher'] = 1 if 'kosher' in text_lower else 0
    features['is_non_gmo'] = 1 if any(phrase in text_lower for phrase in ['non-gmo', 'non gmo']) else 0

    # Premium & marketing features
    features['has_premium_words'] = 1 if any(word in text_lower for word in
                                             ['premium', 'gourmet', 'artisan', 'craft', 'specialty']) else 0
    features['has_health_claims'] = 1 if any(word in text_lower for word in
                                             ['healthy', 'nutritious', 'vitamin', 'protein', 'fiber']) else 0
    features['has_origin_info'] = 1 if any(word in text_lower for word in
                                           ['imported', 'italian', 'french', 'mexican']) else 0
    features['has_pack_info'] = 1 if any(word in text_lower for word in
                                         ['pack of', 'case of', 'bulk', 'count']) else 0

    return features


# Load model function
def load_model_artifacts():
    """Load model artifacts"""
    possible_paths = [
        'enhanced_price_predictor.pkl',
        './enhanced_price_predictor.pkl',
        'models/enhanced_price_predictor.pkl'
    ]

    for path in possible_paths:
        try:
            artifacts = joblib.load(path)
            return artifacts, path
        except:
            continue

    return None, None


# Main app (rest of the code remains the same)
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Price Prediction", "About"])

    if app_mode == "Price Prediction":
        run_prediction()
    else:
        show_about()


def run_prediction():
    # Load model first
    artifacts, loaded_path = load_model_artifacts()

    if artifacts is None:
        st.error("""
        ❌ Model file not found! 

        Please ensure 'enhanced_price_predictor.pkl' is in the same directory as this app.
        """)

        uploaded_model = st.file_uploader("Upload your model file manually", type=['pkl'], key="model_uploader")
        if uploaded_model is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(uploaded_model.getvalue())
                    artifacts = joblib.load(tmp_file.name)
                os.unlink(tmp_file.name)
                st.success("✅ Model loaded from uploaded file!")
            except Exception as e:
                st.error(f"Error loading uploaded model: {str(e)}")
                return
        else:
            return
    else:
        st.success(f"✅ Model loaded successfully from {loaded_path}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Product Image")

        uploaded_file = st.file_uploader(
            "Choose a product image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the product packaging",
            key="image_uploader"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Product Image", use_column_width=True)

            # Process image
            with st.spinner("🔍 Analyzing product image..."):
                # Extract text using EasyOCR
                ocr_text = extract_text_from_image(uploaded_file)

                if ocr_text.strip():
                    st.subheader("📝 Extracted Text")
                    st.text_area("OCR Results", ocr_text, height=150, key="ocr_text")

                    # Extract features from OCR text
                    features = extract_features_from_ocr_text(ocr_text)

                    # Show extracted features
                    st.subheader("🎯 Extracted Features")

                    col_feat1, col_feat2 = st.columns(2)

                    with col_feat1:
                        if features['product_name']:
                            st.markdown(f'<div class="feature-item">**Product:** {features["product_name"]}</div>',
                                        unsafe_allow_html=True)
                        if features['weight_value']:
                            st.markdown(
                                f'<div class="feature-item">**Weight:** {features["weight_value"]} {features["weight_unit"]}</div>',
                                unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="feature-item">**Description Length:** {features["description_length"]} chars</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="feature-item">**Bullet Points:** {features["num_bullet_points"]}</div>',
                            unsafe_allow_html=True)

                    with col_feat2:
                        health_flags = []
                        if features['is_organic']: health_flags.append("🌿 Organic")
                        if features['is_gluten_free']: health_flags.append("🚫 Gluten-Free")
                        if features['is_vegan']: health_flags.append("🌱 Vegan")
                        if features['is_kosher']: health_flags.append("✡️ Kosher")
                        if features['is_non_gmo']: health_flags.append("🧬 Non-GMO")

                        if health_flags:
                            st.markdown('<div class="feature-item">**Certifications:**</div>', unsafe_allow_html=True)
                            for flag in health_flags:
                                st.markdown(f'<div class="feature-item">{flag}</div>', unsafe_allow_html=True)

                    # Make prediction
                    if st.button("🚀 Predict Price", type="primary", key="predict_btn"):
                        make_prediction(features, artifacts)

                else:
                    st.warning(
                        "⚠️ No text could be extracted from the image. Please try a clearer image with visible text.")

    with col2:
        st.subheader("ℹ️ How It Works")
        st.markdown("""
        ### 🎯 Our Innovative Approach

        **1. Image Analysis**
        - Upload any product packaging image
        - Advanced OCR extracts all visible text
        - Smart text processing identifies key features

        **2. Feature Extraction**
        - **Product Specifications**: Weight, size, quantity
        - **Health Certifications**: Organic, gluten-free, vegan, etc.
        - **Marketing Signals**: Premium keywords, origin claims
        - **Content Quality**: Description length, bullet points

        **3. AI Price Prediction**
        - Trained on 75,000+ Amazon products
        - Ensemble machine learning model
        - Real-time price estimation
        """)


def make_prediction(features, artifacts):
    """Make price prediction from extracted features"""
    try:
        # Prepare features for prediction
        feature_df = pd.DataFrame([features])

        # Add engineered features
        feature_df['name_length'] = len(features['product_name'])
        feature_df['has_complex_name'] = 1 if feature_df['name_length'].iloc[0] > 30 else 0

        # Standardize weight
        def standardize_weight_single(weight_value, weight_unit):
            if pd.isna(weight_value) or not weight_unit:
                return None
            unit = str(weight_unit).lower()
            if 'ounce' in unit or 'oz' in unit:
                return weight_value
            elif 'pound' in unit or 'lb' in unit:
                return weight_value * 16
            elif 'gram' in unit or 'g' in unit:
                return weight_value / 28.35
            else:
                return weight_value

        feature_df['weight_standardized'] = standardize_weight_single(
            features['weight_value'], features['weight_unit']
        )

        # Add category (simplified)
        feature_df['product_category_encoded'] = 0

        # Select only the features our model expects
        expected_features = artifacts['feature_names']
        for feature in expected_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0

        feature_df = feature_df[expected_features]
        feature_df = feature_df.fillna(0)

        # Make prediction
        scaled_features = artifacts['scaler'].transform(feature_df)
        predicted_price = artifacts['model'].predict(scaled_features)[0]
        predicted_price = max(predicted_price, 0.01)

        # Display prediction
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("💰 Predicted Price")
        st.markdown(f"<h1 style='color: #FF9900; text-align: center;'>${predicted_price:.2f}</h1>",
                    unsafe_allow_html=True)

        # Confidence score
        confidence_factors = [
            min(features['description_length'] / 100, 1.0),
            min(features['num_bullet_points'] / 10, 1.0),
            1.0 if features['weight_value'] else 0.3,
        ]
        confidence_score = np.mean(confidence_factors) * 100

        st.write("### 📊 Prediction Confidence")
        st.progress(int(confidence_score))
        st.write(f"Confidence Score: {confidence_score:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")


def show_about():
    st.subheader("About This Project")
    st.markdown("""
    ## Amazon Hackathon 2025 - Smart Product Pricing

    **Innovative Features:**
    - 📸 **Image-to-Price**: Upload product images for instant price prediction
    - 🔍 **Advanced OCR**: Extract text from product packaging
    - 🤖 **Ensemble AI**: Multiple machine learning models for accuracy
    - 🎯 **Feature Engineering**: Smart extraction of pricing signals

    **Technical Stack:**
    - Python, Streamlit, EasyOCR, Machine Learning
    - XGBoost, LightGBM, Scikit-learn
    - Advanced feature engineering

    **Team:** PricePredict Pro
    """)


if __name__ == "__main__":
    main()