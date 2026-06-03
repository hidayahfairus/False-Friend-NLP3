import pandas as pd
import numpy as np
import torch
import joblib
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data(file_path):
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    df_clean = df.dropna(subset=['Sentence_Clean_Indo', 'Sentence_Clean_Malay'])
    
    indo_data = pd.DataFrame({'text': df_clean['Sentence_Clean_Indo'], 'label': 'Indonesian'})
    malay_data = pd.DataFrame({'text': df_clean['Sentence_Clean_Malay'], 'label': 'Malay'})
    
    return pd.concat([indo_data, malay_data], ignore_index=True)

def main():
    model_name = 'indobenchmark/indobert-base-p1'
    dataset_path = 'preprocessed_false_friends_dataset.csv'
    classifier_file = 'indobert_svc.joblib'
    
    data = load_data(dataset_path)
    
    # 1. Extract Embeddings
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    print("Extracting sentence embeddings (this may take 1-2 minutes on CPU)...")
    
    # Extract one by one to avoid memory limits or deadlocks
    for text in data['text']:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean of hidden states
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        embeddings.append(sentence_embedding)
        
    X = np.array(embeddings)
    y = data['label']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train SVC Classifier
    print("\n--- Training SVC (RBF Kernel) ---")
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Classifier
    print(f"Saving trained classifier to '{classifier_file}'...")
    joblib.dump(clf, classifier_file)
    print("Trained model saved successfully!")

if __name__ == "__main__":
    main()
