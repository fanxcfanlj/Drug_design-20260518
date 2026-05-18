import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore", message="Unable to import Triton")

# Load the Excel file
df = pd.read_csv(r'DNA.csv')
sequences = df['sequence'].tolist()

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("DNABERT-2-117M", trust_remote_code=True)

# Process each sequence
embeddings = []
for dna in sequences:
    inputs = tokenizer(dna, return_tensors='pt')["input_ids"]
    hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

    # Mean pooling
    embedding_mean = torch.mean(hidden_states[0], dim=0).detach().numpy()
    embeddings.append(embedding_mean)

# Convert to numpy array and save
embeddings_array = np.array(embeddings)
np.savez('buchong_embeddings.npz', embeddings=embeddings_array)