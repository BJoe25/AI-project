Project Overview
This repository contains the implementation for a mini project on fine-tuning a large language model (GPT-2) for explaining Swahili proverbs, a local AI application for Kenyan/African language processing. The goal is to demonstrate improved performance over the base model through fine-tuning on a custom dataset.
Data Provenance
The dataset consists of 415 Swahili proverbs and their explanations, collected from public domain sources to ensure no copyright issues. Key sources include:

Swahili Proverbs database from the University of Illinois – an academic resource sharing traditional methali.
African Manners blog – public compilations of cultural sayings.
Other educational sites like Rough Guides and Pristine Trails, which list proverbs as part of cultural heritage.

These are traditional oral traditions in the public domain. No personal data is included, and the dataset was manually curated and anonymized.
The dataset is available as swahili_proverbs.csv with columns: input (proverb), output (explanation), metadata (source note).
Setup

Install dependencies (Python 3.8+): pip install torch transformers datasets evaluate pandas
Clone the repo (optional GitHub: [link if uploaded]).
Download base GPT-2 model via Hugging Face.

How to Run

Prepare Dataset:
Run dataset_prep.py (if separate) or use the inline code to generate swahili_proverbs.csv.


Fine-Tune:
Run the fine-tuning script: python fine_tune.py
Outputs model to ./fine_tuned_gpt2.


Evaluate:
Run the evaluation script: python evaluate.py --data_path swahili_proverbs.csv
Outputs BLEU scores and sample comparisons to results/eval_results.json.


Example Usage:
Load the tuned model: Python from transformers import pipeline
generator = pipeline("text-generation", model="./fine_tuned_gpt2")
print(generator("Methali: Baada ya dhiki faraja.\nMaelezo:")[0]["generated_text"])
