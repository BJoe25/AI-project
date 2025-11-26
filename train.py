
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',  # Where checkpoints and logs are saved
    eval_strategy="epoch",   # Evaluate at the end of each epoch
    learning_rate=2e-5,      # Typical starting learning rate for BERT
    per_device_train_batch_size=16,  # Batch size per GPU/CPU for training
    per_device_eval_batch_size=64,   # Batch size per GPU/CPU for evaluation
    num_train_epochs=3,      # Total number of epochs
    weight_decay=0.01,       # L2 regularization
)
