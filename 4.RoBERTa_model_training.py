import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

# Load data 
train_file_path = '/{add directory}/train-00000-of-00001.parquet'
val_file_path = '/{add directory}/validation-00000-of-00001.parquet'
test_file_path = '/{add directory}/test-00000-of-00001.parquet'

train_df = pd.read_parquet(train_file_path)
val_df = pd.read_parquet(val_file_path)
test_df = pd.read_parquet(test_file_path)

train_texts = train_df['text'].tolist()
train_labels = train_df['labels'].tolist()

# Vectorized multi-hot encoding
def vectorized_multi_hot_encode(labels_list, num_classes=28):
    encoded = np.zeros((len(labels_list), num_classes), dtype=np.float32)
    for i, labels in enumerate(labels_list):
        encoded[i, labels] = 1.0
    return torch.tensor(encoded)

train_labels = vectorized_multi_hot_encode(train_labels, num_classes=28)

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Define the SentimentDataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Define the emotion labels based on the GoEmotions dataset
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutrality'
]

# Set the threshold for neutral
neutral_threshold = 0.70

# Custom compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = torch.sigmoid(torch.tensor(preds)).cpu().numpy()

    # Convert probabilities to emotion labels with the specified thresholds
    def map_to_emotions(row):
        neutral_prob = row[-1]  # Neutrality is the last class
        if neutral_prob >= neutral_threshold:
            return 'neutrality'
        else:
            non_neutral_probs = row[:-1]
            max_emotion_index = np.argmax(non_neutral_probs)
            return emotion_labels[max_emotion_index]

    # Apply the mapping function to all predictions
    predicted_labels = [map_to_emotions(row) for row in preds]

    # Convert the predicted labels and actual labels back to multi-hot encoding for metric computation
    predicted_multi_hot = np.zeros_like(preds)
    actual_multi_hot = np.zeros_like(preds)

    for i, label in enumerate(predicted_labels):
        predicted_multi_hot[i, emotion_labels.index(label)] = 1
    
    for i, label in enumerate(labels):
        actual_multi_hot[i] = label  

    accuracy = accuracy_score(actual_multi_hot, predicted_multi_hot)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_multi_hot, predicted_multi_hot, average='samples')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }

# K-Fold Cross Validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize a list to store the results for each fold
cv_results = []

# Ensure model uses GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold, (train_index, val_index) in enumerate(kf.split(train_texts)):
    print(f"\nTraining fold {fold + 1}/{k}...")
    
    # Create datasets for the current fold
    train_fold_texts = [train_texts[i] for i in train_index]
    train_fold_labels = train_labels[train_index]
    val_fold_texts = [train_texts[i] for i in val_index]
    val_fold_labels = train_labels[val_index]

    train_dataset = SentimentDataset(train_fold_texts, train_fold_labels)
    val_dataset = SentimentDataset(val_fold_texts, val_fold_labels)

    # Load a fresh model for each fold
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=28).to(device)

    # Set up training arguments with mixed precision and GPU
    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold + 1}',
        num_train_epochs=2,  # Adjust epochs as needed 
        per_device_train_batch_size=16,  # Adjust batch size as needed
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs_fold_{fold + 1}',
        logging_steps=10,
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision only if GPU is available
        evaluation_strategy="steps",
        dataloader_num_workers=2,  # Use multiple workers for data loading
        gradient_accumulation_steps=2,
        no_cuda=not torch.cuda.is_available(),  # Ensure correct device usage
        metric_for_best_model="f1_score",  # Track the best model based on f1_score
        load_best_model_at_end=True,
    )

    # Initialize Trainer for the current fold
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Pass the custom compute metrics function
    )

    # Train the model
    trainer.train()

    # Evaluate the model on validation set
    validation_results = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Results for fold {fold + 1}: {validation_results}")

    # Store the results
    cv_results.append(validation_results)

# After k-fold cross-validation, summarize the results
avg_accuracy = np.mean([result['eval_accuracy'] for result in cv_results])
avg_precision = np.mean([result['eval_precision'] for result in cv_results])
avg_recall = np.mean([result['eval_recall'] for result in cv_results])
avg_f1 = np.mean([result['eval_f1_score'] for result in cv_results])

print("\nCross-Validation Results:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")

# Evaluate the final model on the test set
test_dataset = SentimentDataset(test_texts, test_labels)
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Results:", test_results)

# Save the final model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./tokenizer')
