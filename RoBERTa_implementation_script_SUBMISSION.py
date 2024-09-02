import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import os

# Set the directories for the model and tokenizer
model_dir = '/{add directory}/model'
tokenizer_dir = '/{add directory}/tokenizer'

# Load the model and tokenizer
model = RobertaForSequenceClassification.from_pretrained(model_dir)
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_dir)

# Ensure model uses GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load CSV file with comments only
comments_file_path = '/{add directory}/unique_comments_df.csv'  
comments_df = pd.read_csv(comments_file_path)

# Extract the comments from the self_text column
texts = comments_df['self_text'].tolist()

# Ensure all entries in texts are strings, filtering out non-string entries
texts = [str(text) for text in texts if isinstance(text, (str, int, float))]

# Split texts into smaller batches
batch_size = 16  

# Define the number of batches after which to save progress/ I found a good sweetspot at 1000
save_every_n_batches = 1000

# Define the output file path
output_file_path = '/{add directory}/comments_with_prob_predictions.csv'


# Process in smaller batches to avoid OOM error
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]

    # Tokenize the texts
    inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    # Move inputs to the correct device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Put the model in evaluation mode
    model.eval()

    # Get predictions for the batch
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()
        predicted_labels_df = pd.DataFrame(probs, columns=[f'class_{j}' for j in range(probs.shape[1])])

        # Extract corresponding rows from comments_df
        temp_comments_df = comments_df.iloc[i:i+batch_size].reset_index(drop=True)

        # Concatenate the predictions with the sampled data
        combined_df = pd.concat([temp_comments_df, predicted_labels_df], axis=1)

        # Append the results to the CSV file
        combined_df.to_csv(output_file_path, mode='a', header=not os.path.exists(output_file_path), index=False)

        # Clear memory cache after each batch
        torch.cuda.empty_cache()

    # Print progress message
    if (i // batch_size + 1) % save_every_n_batches == 0:
        print(f"Saved progress up to batch {i//batch_size + 1} to {output_file_path}")

print(f"Final predictions saved to {output_file_path}")
