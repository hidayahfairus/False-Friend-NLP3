import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Malay-Indo False Friend AI", page_icon="🇲🇾", layout="centered")

# App Header
st.title("🇲🇾 Malay vs 🇮🇩 Indonesian AI")
st.markdown("### False Friend Disambiguator (BERT)")
st.write("Enter a sentence containing a 'false friend' word to identify its linguistic context.")

# Configuration
MODEL_NAME = 'indobenchmark/indobert-base-p1'
CLASSIFIER_FILE = 'indobert_svc.joblib'

# Cache the model to avoid reloading on every interaction
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    if os.path.exists(CLASSIFIER_FILE):
        clf = joblib.load(CLASSIFIER_FILE)
    else:
        st.error(f"Classifier file '{CLASSIFIER_FILE}' not found. Please run the training script first.")
        clf = None
        
    return tokenizer, model, clf, device

tokenizer, model, clf, device = load_models()

# Input Section
user_input = st.text_input("Enter your sentence:", "Saya memandu kereta baru ke pejabat.")

if st.button("Predict Language"):
    if user_input and clf and model:
        with st.spinner("Analyzing context..."):
            # Get BERT embedding
            inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Predict
            prediction = clf.predict(emb)[0]
            probs = clf.predict_proba(emb)[0]
            
            # Show Results
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == "Malay":
                    st.success(f"Language: **{prediction}** 🇲🇾")
                else:
                    st.info(f"Language: **{prediction}** 🇮🇩")
            
            with col2:
                st.metric("Confidence", f"{max(probs):.2%}")
                
            st.progress(float(max(probs)))
            
            # Explanation
            if "pejabat" in user_input.lower():
                st.info("**Context Note**: 'Pejabat' in Malay usually means 'Office', whereas in Indonesian it means 'Official'.")
            elif "kereta" in user_input.lower():
                st.info("**Context Note**: 'Kereta' in Malay usually means 'Car', whereas in Indonesian it means 'Train'.")
            elif "bisa" in user_input.lower():
                st.info("**Context Note**: 'Bisa' in Malay means 'Venom', whereas in Indonesian it usually means 'Can/Able'.")
    else:
        st.warning("Please enter a sentence to test.")

# Footer
st.divider()
st.caption("Developed for STINK3114 NLP Assignment - Universiti Utara Malaysia")
