import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.warning("Google API Key not found. Please set GOOGLE_API_KEY in .env file for Health Bot to work.")

# Set page configuration
st.set_page_config(page_title="Multiple Disease Prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")

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
        return f"Could not generate advice: {e}"

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
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Model Configuration
    selected_model = "gemini-flash-lite-latest"

    tools = [predict_diabetes, predict_heart_disease, predict_parkinsons]

    # Initialize session state for model
    if "current_model" not in st.session_state:
        st.session_state.current_model = selected_model
        st.session_state.chat_session = None
        st.session_state.messages = []

    if "chat_session" not in st.session_state or st.session_state.chat_session is None:
        try:
            if google_api_key:
                system_prompt = "You are a helpful medical assistant. You strictly answer only health-related questions. You can answer general medical questions about any disease, symptoms, or health condition. You also have specialized access to predictive models for Diabetes, Heart Disease, and Parkinson's. Use these specific tools ONLY when the user asks for a risk assessment for these three diseases and provides the necessary clinical data. For other health questions, answer using your general medical knowledge."
                model = genai.GenerativeModel(selected_model, system_instruction=system_prompt, tools=tools)
                st.session_state.chat_session = model.start_chat(enable_automatic_function_calling=True)
            else:
                st.warning("API Key missing.")
        except Exception as e:
            st.error(f"Failed to initialize chat: {e}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your health concern?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            if st.session_state.chat_session:
                response = st.session_state.chat_session.send_message(prompt)
                response_text = response.text
                
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.error("Chat session not initialized. Please check API Key.")
        except Exception as e:
            error_str = str(e)
            response_text = f"Error: {e}"
            if "429" in error_str:
                response_text = "⚠️ **Quota Exceeded**: Please select a different model."
            
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # Code for Prediction
    diab_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin),
                          float(BMI), float(DiabetesPedigreeFunction), float(Age)]
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
        except ValueError:
            st.error("Please enter valid numerical values.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1 = male; 0 = female)')
    with col3:
        cp = st.text_input('Chest Pain types (0, 1, 2, 3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0, 1, 2)')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0, 1, 2)')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy (0-3)')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg),
                          float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
            heart_prediction = heart_disease_model.predict([user_input])
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
            st.success(heart_diagnosis)
            
            # AI Advice
            input_summary = f"Age: {age}, Cholesterol: {chol}, BP: {trestbps}, Max Heart Rate: {thalach}, Sex: {'Male' if sex=='1' else 'Female'}"
            advice = generate_health_advice("Heart Disease", heart_diagnosis, input_summary)
            st.info("💡 **AI Health Advice**")
            st.markdown(advice)
        except ValueError:
            st.error("Please enter valid numerical values.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                          float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                          float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                          float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
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
        except ValueError:
            st.error("Please enter valid numerical values.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
