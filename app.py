import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Set page configuration
st.set_page_config(page_title="Multiple Disease Prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load environment variables
load_dotenv()

# Configure Gemini API
# Try to get key from Streamlit secrets (deployment) or environment variables (local)
try:
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        google_api_key = os.getenv("GOOGLE_API_KEY")
except (FileNotFoundError, Exception):
    google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.warning("Google API Key not found. Please set GOOGLE_API_KEY in .env file (local) or Streamlit Secrets (cloud).")

# Loading the saved models
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = None
heart_disease_model = None
parkinsons_model = None

try:
    diabetes_model = pickle.load(open(os.path.join(working_dir, 'diabetes_model.sav'), 'rb'))
    heart_disease_model = pickle.load(open(os.path.join(working_dir, 'heart_disease_model.sav'), 'rb'))
    parkinsons_model = pickle.load(open(os.path.join(working_dir, 'parkinsons_model.sav'), 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

# Tool Functions for Chatbot
def predict_diabetes(pregnancies: float, glucose: float, blood_pressure: float, skin_thickness: float, insulin: float, bmi: float, diabetes_pedigree_function: float, age: float):
    """
    Predicts if a person has diabetes based on clinical features.
    
    Args:
        pregnancies: Number of times pregnant
        glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
        blood_pressure: Diastolic blood pressure (mm Hg)
        skin_thickness: Triceps skin fold thickness (mm)
        insulin: 2-Hour serum insulin (mu U/ml)
        bmi: Body mass index (weight in kg/(height in m)^2)
        diabetes_pedigree_function: Diabetes pedigree function
        age: Age (years)
        
    Returns:
        str: "The person is diabetic" or "The person is not diabetic"
    """
    try:
        user_input = [float(pregnancies), float(glucose), float(blood_pressure), float(skin_thickness), float(insulin), float(bmi), float(diabetes_pedigree_function), float(age)]
        prediction = diabetes_model.predict([user_input])
        return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'
    except Exception as e:
        return f"Error in diabetes prediction: {str(e)}"

def predict_heart_disease(age: float, sex: int, cp: int, trestbps: float, chol: float, fbs: int, restecg: int, thalach: float, exang: int, oldpeak: float, slope: int, ca: int, thal: int):
    """
    Predicts if a person has heart disease.
    
    Args:
        age: Age in years
        sex: 1 = male; 0 = female
        cp: Chest pain type (0, 1, 2, 3)
        trestbps: Resting blood pressure (in mm Hg on admission to the hospital)
        chol: Serum cholestoral in mg/dl
        fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
        restecg: Resting electrocardiographic results (0, 1, 2)
        thalach: Maximum heart rate achieved
        exang: Exercise induced angina (1 = yes; 0 = no)
        oldpeak: ST depression induced by exercise relative to rest
        slope: The slope of the peak exercise ST segment (0, 1, 2)
        ca: Number of major vessels (0-3) colored by flourosopy
        thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
        
    Returns:
        str: "The person is having heart disease" or "The person does not have any heart disease"
    """
    try:
        user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
        prediction = heart_disease_model.predict([user_input])
        return 'The person is having heart disease' if prediction[0] == 1 else 'The person does not have any heart disease'
    except Exception as e:
        return f"Error in heart disease prediction: {str(e)}"

def predict_parkinsons(fo: float, fhi: float, flo: float, jitter_percent: float, jitter_abs: float, rap: float, ppq: float, ddp: float, shimmer: float, shimmer_db: float, apq3: float, apq5: float, apq: float, dda: float, nhr: float, hnr: float, rpde: float, dfa: float, spread1: float, spread2: float, d2: float, ppe: float):
    """
    Predicts if a person has Parkinson's disease based on acoustic features.
    
    Args:
        fo: MDVP:Fo(Hz) - Average vocal fundamental frequency
        fhi: MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
        flo: MDVP:Flo(Hz) - Minimum vocal fundamental frequency
        jitter_percent: MDVP:Jitter(%)
        jitter_abs: MDVP:Jitter(Abs)
        rap: MDVP:RAP
        ppq: MDVP:PPQ
        ddp: Jitter:DDP
        shimmer: MDVP:Shimmer
        shimmer_db: MDVP:Shimmer(dB)
        apq3: Shimmer:APQ3
        apq5: Shimmer:APQ5
        apq: MDVP:APQ
        dda: Shimmer:DDA
        nhr: NHR
        hnr: HNR
        rpde: RPDE
        dfa: DFA
        spread1: spread1
        spread2: spread2
        d2: D2
        ppe: PPE
        
    Returns:
        str: "The person has Parkinson's disease" or "The person does not have Parkinson's disease"
    """
    try:
        user_input = [float(fo), float(fhi), float(flo), float(jitter_percent), float(jitter_abs), float(rap), float(ppq), float(ddp), float(shimmer), float(shimmer_db), float(apq3), float(apq5), float(apq), float(dda), float(nhr), float(hnr), float(rpde), float(dfa), float(spread1), float(spread2), float(d2), float(ppe)]
        prediction = parkinsons_model.predict([user_input])
        return "The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease"
    except Exception as e:
        return f"Error in parkinsons prediction: {str(e)}"

def generate_health_advice(disease_name, prediction_result, input_data_summary):
    """
    Generates health advice using Gemini based on the prediction result.
    """
    if not google_api_key:
        return "Please set GOOGLE_API_KEY to receive AI health advice."
    
    try:
        # Use the selected model from session state if available, otherwise default
        model_name = st.session_state.get("current_model", "gemini-flash-lite-latest")
        
        system_prompt = "You are a helpful medical assistant. Provide concise, empathetic, and practical health advice based on the user's condition and risk factors. Do not diagnose, but suggest lifestyle changes or next steps."
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
        
        prompt = f"""
        Context: The user has just run a {disease_name} risk assessment.
        Result: {prediction_result}
        Input Vitals/Data: {input_data_summary}
        
        Please provide:
        1. A brief explanation of what this result might mean.
        2. 3-4 specific, actionable tips based on their provided data (e.g., BMI, glucose, etc).
        3. A disclaimer that this is not a medical diagnosis and they should consult a doctor.
        Keep it under 150 words.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return response.text
    except Exception as e:
        return f"Could not generate advice: {e}"

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gemini_history" not in st.session_state:
        st.session_state.gemini_history = []

def display_chat_interface():
    st.markdown("---")
    st.subheader("💬 Chat with Health Bot")
    
    initialize_chat_history()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    selected_model = "gemini-flash-lite-latest"
    tools = [predict_diabetes, predict_heart_disease, predict_parkinsons]
    
    if google_api_key:
        try:
            system_prompt = "You are a helpful medical assistant. You strictly answer only health-related questions. You can answer general medical questions about any disease, symptoms, or health condition. You also have specialized access to predictive models for Diabetes, Heart Disease, and Parkinson's. Use these specific tools ONLY when the user asks for a risk assessment for these three diseases and provides the necessary clinical data. For other health questions, answer using your general medical knowledge. If someone asks who created you or who designed you, reply that you were designed by Huzaif, an AI engineer for medical purposes. Answer in a professional way."
            model = genai.GenerativeModel(selected_model, system_instruction=system_prompt, tools=tools)
            chat_session = model.start_chat(history=st.session_state.gemini_history, enable_automatic_function_calling=True)
        except Exception as e:
            st.error(f"Failed to initialize chat: {e}")
            return
    else:
        st.warning("API Key missing.")
        return

    if prompt := st.chat_input("Ask a follow-up question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = chat_session.send_message(prompt)
            response_text = response.text
            
            # Update history manually
            st.session_state.gemini_history.append({"role": "user", "parts": [prompt]})
            st.session_state.gemini_history.append({"role": "model", "parts": [response_text]})

            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                st.error("⚠️ **Quota Exceeded**: Please select a different model or try again later.")
            else:
                st.error(f"Error: {e}")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Health Bot',
                            'Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons=['chat-dots-fill', 'activity', 'heart', 'person'],
                           default_index=0)

# Health Bot Page
if selected == 'Health Bot':
    st.title("Health Bot")
    display_chat_interface()

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0)
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0)
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0)
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, format="%.1f")
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, format="%.3f")
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, step=1)

    # Code for Prediction
    diab_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
            diab_prediction = diabetes_model.predict([user_input])
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
            st.success(diab_diagnosis)
            
            # AI Advice
            input_summary = f"Glucose: {Glucose}, BMI: {BMI}, Age: {Age}, BP: {BloodPressure}, Insulin: {Insulin}"
            advice = generate_health_advice("Diabetes", diab_diagnosis, input_summary)
            st.info("💡 **AI Health Advice**")
            st.markdown(advice)
            
            # Add context to chat history so user can follow up
            initialize_chat_history()
            st.session_state.gemini_history.append({"role": "user", "parts": [f"I have just done a Diabetes check. Result: {diab_diagnosis}. Stats: {input_summary}. Advice given: {advice}"]})
            st.session_state.gemini_history.append({"role": "model", "parts": ["I understand. I'm here to help if you have questions about your diabetes results or the advice."]})
            st.session_state.messages.append({"role": "assistant", "content": "I've analyzed your results. Feel free to ask any follow-up questions below!"})

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Always show chat interface at bottom
    display_chat_interface()

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, step=1)
    with col2:
        sex = st.number_input('Sex (1 = male; 0 = female)', min_value=0, max_value=1, step=1)
    with col3:
        cp = st.number_input('Chest Pain types (0, 1, 2, 3)', min_value=0, max_value=3, step=1)
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0)
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0)
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', min_value=0, max_value=1, step=1)
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results (0, 1, 2)', min_value=0, max_value=2, step=1)
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0)
    with col3:
        exang = st.number_input('Exercise Induced Angina (1 = yes; 0 = no)', min_value=0, max_value=1, step=1)
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', format="%.1f")
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment (0, 1, 2)', min_value=0, max_value=2, step=1)
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy (0-3)', min_value=0, max_value=3, step=1)
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', min_value=0, max_value=2, step=1)

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]
            heart_prediction = heart_disease_model.predict([user_input])
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
            st.success(heart_diagnosis)
            
            # AI Advice
            input_summary = f"Age: {age}, Cholesterol: {chol}, BP: {trestbps}, Max Heart Rate: {thalach}, Sex: {'Male' if sex==1 else 'Female'}"
            advice = generate_health_advice("Heart Disease", heart_diagnosis, input_summary)
            st.info("💡 **AI Health Advice**")
            st.markdown(advice)

            # Add context to chat history
            initialize_chat_history()
            st.session_state.gemini_history.append({"role": "user", "parts": [f"I have just done a Heart Disease check. Result: {heart_diagnosis}. Stats: {input_summary}. Advice given: {advice}"]})
            st.session_state.gemini_history.append({"role": "model", "parts": ["I understand. I'm here to help if you have questions about your heart health results."]})
            st.session_state.messages.append({"role": "assistant", "content": "I've analyzed your results. Feel free to ask any follow-up questions below!"})

        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    # Always show chat interface
    display_chat_interface()

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', format="%.3f")
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', format="%.3f")
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', format="%.3f")
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', format="%.5f")
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', format="%.5f")
    with col1:
        RAP = st.number_input('MDVP:RAP', format="%.5f")
    with col2:
        PPQ = st.number_input('MDVP:PPQ', format="%.5f")
    with col3:
        DDP = st.number_input('Jitter:DDP', format="%.5f")
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', format="%.5f")
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', format="%.3f")
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', format="%.5f")
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', format="%.5f")
    with col3:
        APQ = st.number_input('MDVP:APQ', format="%.5f")
    with col4:
        DDA = st.number_input('Shimmer:DDA', format="%.5f")
    with col5:
        NHR = st.number_input('NHR', format="%.5f")
    with col1:
        HNR = st.number_input('HNR', format="%.3f")
    with col2:
        RPDE = st.number_input('RPDE', format="%.5f")
    with col3:
        DFA = st.number_input('DFA', format="%.5f")
    with col4:
        spread1 = st.number_input('spread1', format="%.5f")
    with col5:
        spread2 = st.number_input('spread2', format="%.5f")
    with col1:
        D2 = st.number_input('D2', format="%.5f")
    with col2:
        PPE = st.number_input('PPE', format="%.5f")

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                          APQ3, APQ5, APQ, DDA, NHR, HNR,
                          RPDE, DFA, spread1, spread2, D2, PPE]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)
            
            # AI Advice
            input_summary = f"Vocal Fo: {fo}, Jitter%: {Jitter_percent}, Shimmer: {Shimmer}, HNR: {HNR}"
            advice = generate_health_advice("Parkinson's Disease", parkinsons_diagnosis, input_summary)
            st.info("💡 **AI Health Advice**")
            st.markdown(advice)
            
            # Add context to chat history
            initialize_chat_history()
            st.session_state.gemini_history.append({"role": "user", "parts": [f"I have just done a Parkinson's check. Result: {parkinsons_diagnosis}. Stats: {input_summary}. Advice given: {advice}"]})
            st.session_state.gemini_history.append({"role": "model", "parts": ["I understand. I'm here to help if you have questions about your Parkinson's assessment."]})
            st.session_state.messages.append({"role": "assistant", "content": "I've analyzed your results. Feel free to ask any follow-up questions below!"})

        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    # Always show chat interface
    display_chat_interface()
