import pickle
import os
import sklearn
import sys

with open('model_info.txt', 'w') as log_file:
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    print(f"Scikit-learn version: {sklearn.__version__}")

    files = ['diabetes_model.sav', 'heart_disease_model.sav', 'parkinsons_model.sav']

    print("-" * 30)

    for f in files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            try:
                model = pickle.load(open(f, 'rb'))
                print(f"Type: {type(model)}")
                if hasattr(model, 'n_features_in_'):
                    print(f"Number of features: {model.n_features_in_}")
                if hasattr(model, 'feature_names_in_'):
                    print(f"Feature names: {model.feature_names_in_}")
            except Exception as e:
                print(f"Error loading {f}: {e}")
        else:
            print(f"File {f} not found.")
        print("-" * 30)

    sys.stdout = original_stdout
