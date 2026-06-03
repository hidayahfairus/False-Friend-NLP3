import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
import joblib
import os
import numpy as np
from colorama import init, Fore, Style
import pandas as pd
from tqdm import tqdm

# Initialize Colorama
init(autoreset=True)

from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

# Configuration
MODEL_NAME = 'indobenchmark/indobert-base-p1'
CLASSIFIER_FILE = 'indobert_svc.joblib'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_demo():
    print(f"\n{Fore.MAGENTA}==================================================")
    print(f"{Fore.MAGENTA}   MALAY-INDO FALSE FRIEND DISAMBIGUATOR (BERT)   ")
    print(f"{Fore.MAGENTA}==================================================")
    
    if not os.path.exists(CLASSIFIER_FILE):
        print(f"{Fore.RED}Error: Classifier file '{CLASSIFIER_FILE}' not found.")
        print(f"{Fore.YELLOW}Please train it first.")
        return
        
    print(f"{Fore.CYAN}Loading tokenizer, IndoBERT model, and SVC classifier...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    clf = joblib.load(CLASSIFIER_FILE)
    print(f"{Fore.GREEN}Loaded model and classifier successfully!")

    print(f"\n{Fore.CYAN}Ready! Type a sentence below to test.")
    print(f"{Fore.WHITE}Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input(f"{Fore.YELLOW}Sentence > {Style.RESET_ALL}").strip()
            
            if user_input.lower() == 'exit':
                print(f"{Fore.CYAN}Goodbye!")
                break
            
            if not user_input:
                continue

            # Prediction
            inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
            prediction = clf.predict(emb)[0]
            probs = clf.predict_proba(emb)[0]
            
            # Formatting Output
            color = Fore.GREEN if prediction == 'Malay' else Fore.BLUE
            print(f"  {Fore.WHITE}Prediction: {color}{prediction}")
            print(f"  {Fore.WHITE}Confidence: {Fore.WHITE}{max(probs):.2%}")
            print("-" * 30)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}")

if __name__ == "__main__":
    run_demo()
