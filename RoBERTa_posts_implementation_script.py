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

# Load your new dataset from a CSV file
new_data_file_path = '/{add directory}/relevant_posts.csv'  
posts_df = pd.read_csv(new_data_file_path)

# Handle potential blanks in `post_title` and `post_self_text`
posts_df['post_title'].fillna('', inplace=True)
posts_df['post_self_text'].fillna('', inplace=True)

# Combine `post_title` and `post_self_text` into a single text column
posts_df['combined_text'] = posts_df['post_title'] + ' ' + posts_df['post_self_text']

# Strip any leading or trailing whitespace from the combined text
posts_df['combined_text'] = posts_df['combined_text'].str.strip()

# Ensure all entries in texts are strings
texts = posts_df['combined_text'].tolist()

# Split texts into smaller batches
batch_size = 16 

# Define the number of batches after which to save progress/ I found a good sweetspot at 1000
save_every_n_batches = 1000  

# Define the output file path
output_file_path = '/{add directory}/posts_with_prob_predictions.csv'  

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

        # Extract corresponding rows from posts_df
        temp_posts_df = posts_df.iloc[i:i+batch_size].reset_index(drop=True)

        # Concatenate the predictions with the original data
        combined_df = pd.concat([temp_posts_df, predicted_labels_df], axis=1)

        # Append the results to the CSV file
        combined_df.to_csv(output_file_path, mode='a', header=not os.path.exists(output_file_path), index=False)

        # Clear memory cache after each batch
        torch.cuda.empty_cache()

    # Print progress message
    if (i // batch_size + 1) % save_every_n_batches == 0:
        print(f"Saved progress up to batch {i//batch_size + 1} to {output_file_path}")

print(f"Final predictions saved to {output_file_path}")
