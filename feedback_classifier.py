import streamlit as st
import pandas as pd
import time
import ibm_boto3
import os
from dotenv import load_dotenv
from botocore.client import Config
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import types

# Load environment variables from .env file
load_dotenv()

# IBM Cloud Configuration
CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("IBM_API_KEY")  # From .env file
}
PROJECT_ID = os.getenv("IBM_PROJECT_ID")  # From .env file
MODEL_ID = ModelTypes.FLAN_T5_XXL


def load_data_from_cos(bucket_name, object_key):
    """Load data from IBM Cloud Object Storage"""
    def __iter__(self): return 0
    
    cos_client = ibm_boto3.client(
        service_name='s3',
        ibm_api_key_id=os.getenv("COS_API_KEY"),  # From .env file
        ibm_auth_endpoint="https://iam.cloud.ibm.com/identity/token",
        config=Config(signature_version='oauth'),
        endpoint_url='https://s3.us-south.cloud-object-storage.appdomain.cloud'
    )
    
    try:
        body = cos_client.get_object(Bucket=bucket_name, Key=object_key)['Body']
        if not hasattr(body, "__iter__"): 
            body.__iter__ = types.MethodType(__iter__, body)
        return pd.read_csv(body)
    except Exception as e:
        st.error(f"Error loading data from COS: {e}")
        return None

@st.cache_resource
def init_model():
    """Initialize and cache the Watson ML model"""
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.RANDOM_SEED: 33,
        GenParams.MAX_NEW_TOKENS: 5,
        GenParams.MIN_NEW_TOKENS: 1
    }
    return Model(
        model_id=MODEL_ID,
        params=parameters,
        credentials=CREDENTIALS,
        project_id=PROJECT_ID
    )

def prepare_few_shot_examples(train_data, num_examples=5):
    """Create few-shot examples from training data"""
    examples = []
    for i in range(min(num_examples, len(train_data))):
        examples.append(f"Feedback:\t{train_data.values[i][1]}\nTheme:\t{train_data.values[i][2]}\n\n")
    return "".join(examples)

def create_prompt(feedback, instruction, examples):
    """Generate prompt for model inference"""
    return f"{instruction}{examples}Feedback:\t{feedback}\nTheme:\t"

def predict_theme(model, prompt):
    """Get theme prediction from model"""
    try:
        response = model.generate(prompt)
        return response["results"][0]["generated_text"].strip()
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        return None

# Main Application 
def main():
    # Page Configuration
    st.set_page_config(
        page_title="College Feedback Classifier",
        page_icon="🏫",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Custom Styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #808080;
    }
    .header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .result-box {
        background: black;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 5px solid #4facfe;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
    }
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App Header
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; padding:0">🏫 College Feedback Classifier</h1>
        <p style="opacity:0.8; margin-top:0.5rem">AI-powered theme classification for student feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verify credentials
    if not CREDENTIALS["apikey"] or not PROJECT_ID:
        st.error("Missing IBM Cloud credentials. Please check your .env file")
        return
    
    # Load data from IBM Cloud
    bucket_name = os.getenv("COS_BUCKET", "collegefeedbackclassifier-donotdelete-pr-ejz40fmj7wizdg")
    
    with st.spinner("Loading training data from cloud..."):
        train_data = load_data_from_cos(
            bucket_name=bucket_name,
            object_key='feedack_train_1.csv'
        )
    
    with st.spinner("Loading test data from cloud..."):
        test_data = load_data_from_cos(
            bucket_name=bucket_name,
            object_key='feedack_test_1.csv'
        )
    
    if train_data is None or test_data is None:
        st.error("Failed to load data. Check connection and credentials.")
        return
    
    # Prepare prompts and model
    with st.spinner("Preparing AI model..."):
        instruction = """Find the theme of the college student feedback.
Choose the theme from the following list:
'Academics', 'Facilities', 'Administration'.\n\n"""
        
        few_shot_examples = prepare_few_shot_examples(train_data, num_examples=5)
        model = init_model()
    
    # Section 1: Test Results
    st.subheader("🧪 Model Test Results")
    st.markdown(f"Evaluated on first 10 samples from test dataset")
    
    # Run predictions for first 10 test samples
    test_results = []
    feedbacks = []
    actual_themes = []
    
    for i in range(min(10, len(test_data))):
        feedback = test_data.values[i][1]
        actual_theme = test_data.values[i][2]
        prompt = create_prompt(feedback, instruction, few_shot_examples)
        
        with st.spinner(f"Processing test sample {i+1}/10..."):
            predicted_theme = predict_theme(model, prompt)
            time.sleep(0.6)  # Rate limiting
            
        test_results.append({
            "Feedback": feedback,
            "Actual Theme": actual_theme,
            "Predicted Theme": predicted_theme,
            "Match": "✅" if actual_theme == predicted_theme else "❌"
        })
        feedbacks.append(feedback)
        actual_themes.append(actual_theme)
    
    # Show results table
    results_df = pd.DataFrame(test_results)
    st.dataframe(results_df.style.applymap(
        lambda x: "background-color: #e6f7ff" if x == "✅" else "background-color: #ffebee", 
        subset=["Match"]
    ))
    
    # Calculate accuracy
    accuracy = sum([1 for r in test_results if r["Match"] == "✅"]) / len(test_results)
    st.metric("Test Accuracy", f"{accuracy:.0%}")
    
    # Section 2: User Feedback Classification
    st.divider()
    st.subheader("🔍 Classify Your Feedback")
    
    # User input
    user_feedback = st.text_area(
        "Enter student feedback:",
        placeholder="e.g., 'The library hours should be extended during exam season'",
        height=150,
        key="user_input"
    )
    
    # Classification button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_btn = st.button("Analyze Feedback", use_container_width=True)
    
    # Process user input
    if classify_btn:
        if not user_feedback.strip():
            st.warning("Please enter feedback text to classify")
        else:
            with st.spinner("Analyzing feedback..."):
                prompt = create_prompt(user_feedback, instruction, few_shot_examples)
                theme = predict_theme(model, prompt)
                time.sleep(0.6)  # Rate limiting
                
            if theme:
                # Display results
                theme_emojis = {
                    "Academics": "📚",
                    "Facilities": "🏢",
                    "Administration": "💼"
                }
                emoji = theme_emojis.get(theme, "❓")
                
                st.divider()
                st.subheader("Classification Result")
                st.markdown(f"""
                <div class="result-box">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <div style="font-size: 3rem;">{emoji}</div>
                        <div>
                            <h3 style="margin: 0; color: #1e3a8a;">{theme}</h3>
                            <p style="margin: 5px 0 0; color: #555;">Theme classification</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show technical details
                with st.expander("View prompt details"):
                    st.markdown("**Generated Prompt:**")
                    st.code(prompt)
                    st.markdown(f"**Model Response:** `{theme}`")

if __name__ == "__main__":
    main()