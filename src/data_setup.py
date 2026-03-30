import pandas as pd
import json
import os

# Create directories
os.makedirs('dataset_files/extractive', exist_ok=True)
os.makedirs('dataset_files/abstractive', exist_ok=True)

def process_cnn():
    df = pd.read_csv('datasets/cnn_dailymail/sample.csv')
    data = []
    for _, row in df.iterrows():
        # Truncate article to ~300 words to fit in context for few-shot
        article = " ".join(str(row['article']).split()[:200])
        summary = str(row['highlights']).strip()
        data.append({"input": article, "output": summary})
    
    with open('dataset_files/extractive/cnn.json', 'w') as f:
        json.dump(data, f, indent=2)

def process_xsum():
    df = pd.read_csv('datasets/xsum/sample.csv')
    data = []
    for _, row in df.iterrows():
        # Truncate article to ~300 words
        article = " ".join(str(row['document']).split()[:200])
        summary = str(row['summary']).strip()
        data.append({"input": article, "output": summary})
        
    with open('dataset_files/abstractive/xsum.json', 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    process_cnn()
    process_xsum()
    print("Datasets formatted and saved to dataset_files/")
