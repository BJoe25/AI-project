import os
import json
import argparse
from typing import List, Tuple

import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2TokenizerFast
import evaluate
import pandas as pd

def load_dataset_from_csv(path: str) -> List[Tuple[str, str]]:
    """
    Expects CSV with either:
      - columns 'input' and 'output'
      - OR a single column 'text' with lines in format "Methali: <proverb>\nMaelezo: <explanation>"
    Returns list of (proverb, explanation)
    """
    df = pd.read_csv(path)
    pairs = []
    if {"input", "output"}.issubset(df.columns):
        for _, row in df.iterrows():
            pairs.append((str(row["input"]).strip(), str(row["output"]).strip()))
        return pairs

    if "text" in df.columns:
        for _, row in df.iterrows():
            txt = str(row["text"])
            # try to split on the "Maelezo:" marker
            if "Maelezo:" in txt:
                parts = txt.split("Maelezo:")
                prompt_part = parts[0].replace("Methali:", "").strip()
                explanation = parts[1].strip()
                pairs.append((prompt_part, explanation))
            else:
                # fallback - use whole text as input and empty reference
                pairs.append((txt.strip(), ""))
        return pairs

    # otherwise try interpreting the first two columns as proverb & explanation
    if df.shape[1] >= 2:
        for _, row in df.iterrows():
            p = str(row.iloc[0]).strip()
            e = str(row.iloc[1]).strip()
            pairs.append((p, e))
        return pairs

    raise ValueError("CSV format unrecognized. Provide columns 'input'/'output' or 'text'.")

def load_pairs_from_module() -> List[Tuple[str, str]]:
    """
    Try to import `pairs` from fine_tune.py (so that file should be present in repo).
    `pairs` should be a list of (proverb, explanation).
    """
    try:
        import fine_tune
        if hasattr(fine_tune, "pairs"):
            return fine_tune.pairs
        else:
            raise ImportError("fine_tune.py found but no 'pairs' variable inside.")
    except Exception as e:
        raise ImportError(f"Could not import pairs from fine_tune.py: {e}")

def build_prompt(proverb: str) -> str:
    # Match the format used in training: "Methali: ...\nMaelezo: "
    return f"Methali: {proverb}\nMaelezo:"

def extract_generated_explanation(generated_text: str) -> str:
    # Remove any preceding prompt and return the text after the 'Maelezo:' marker
    if "Maelezo:" in generated_text:
        return generated_text.split("Maelezo:")[-1].strip()
    # fallback: return the generated string
    return generated_text.strip()

def generate_for_prompts(pipe, prompts: List[str], max_new_tokens: int = 60) -> List[str]:
    generated = []
    for p in prompts:
        out = pipe(p, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1)[0]["generated_text"]
        gen = extract_generated_explanation(out)
        generated.append(gen)
    return generated

def compute_bleu(preds: List[str], refs: List[str]):
    bleu = evaluate.load("bleu")
    # predictions: list of token lists
    predictions = [p.split() for p in preds]
    # references: list of list of token lists (one reference per sample)
    references = [[r.split()] for r in refs]
    result = bleu.compute(predictions=predictions, references=references)
    return result

def main(args):
    # 1. Load dataset (prefer CSV; otherwise import pairs from fine_tune.py)
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading dataset from {args.data_path}")
        pairs = load_dataset_from_csv(args.data_path)
    else:
        print("No data file found at --data_path. Trying to import `pairs` from fine_tune.py (must exist).")
        try:
            pairs = load_pairs_from_module()
            print(f"Loaded {len(pairs)} samples from fine_tune.py")
        except ImportError as e:
            print("ERROR: Couldn't find dataset. Please either:")
            print("  1) place a CSV at './data/swahili_proverbs.csv' with columns 'input' and 'output' (or 'text'), OR")
            print("  2) ensure `fine_tune.py` with `pairs` list is in the repo so evaluate.py can import it.")
            raise SystemExit(str(e))

    # Build prompts and references (we expect pairs to be tuples (proverb, explanation) or already in training format)
    proverbs = []
    references = []
    for a, b in pairs:
        # If the pair looks like "Methali: ..." stored already, strip possible prefixes
        if isinstance(a, str) and a.strip().startswith("Methali:"):
            # find the proverb line
            txt = a.strip()
            # Attempt to extract the content after "Methali:"
            proverb = txt.replace("Methali:", "").strip()
        else:
            proverb = str(a).strip()
        proverbs.append(proverb)
        references.append(str(b).strip())

    # Limit evaluation size if requested
    N = args.num_samples if args.num_samples and args.num_samples > 0 else len(proverbs)
    proverbs = proverbs[:N]
    references = references[:N]
    prompts = [build_prompt(p) for p in proverbs]

    # 2. Setup device (GPU if available)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device==0 else 'cpu'}")

    # 3. Load base model pipeline
    print("Loading base GPT-2 model...")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(args.base_model_name)
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer, device=device)

    # 4. Load fine-tuned model pipeline
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Fine-tuned model directory not found at {args.model_dir}")
    print(f"Loading fine-tuned model from {args.model_dir} ...")
    tuned_model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    tuned_tokenizer = GPT2TokenizerFast.from_pretrained(args.model_dir)
    tuned_tokenizer.pad_token = tuned_tokenizer.eos_token
    tuned_pipe = pipeline("text-generation", model=tuned_model, tokenizer=tuned_tokenizer, device=device)

    # 5. Generate
    print(f"Generating {len(prompts)} samples with base model...")
    base_outputs = generate_for_prompts(base_pipe, prompts, max_new_tokens=args.max_new_tokens)
    print(f"Generating {len(prompts)} samples with fine-tuned model...")
    tuned_outputs = generate_for_prompts(tuned_pipe, prompts, max_new_tokens=args.max_new_tokens)

    # 6. Compute BLEU
    print("Computing BLEU scores...")
    base_bleu = compute_bleu(base_outputs, references)
    tuned_bleu = compute_bleu(tuned_outputs, references)

    # 7. Print results and sample comparison
    summary = {
        "base_model": args.base_model_name,
        "fine_tuned_model_dir": args.model_dir,
        "num_samples": len(prompts),
        "base_bleu": base_bleu,
        "tuned_bleu": tuned_bleu
    }
    print("\n=== BLEU RESULTS ===")
    print(f"Base GPT-2 BLEU: {base_bleu}")
    print(f"Fine-tuned GPT-2 BLEU: {tuned_bleu}")

    print("\n=== SAMPLE COMPARISONS (first 8) ===")
    for i in range(min(8, len(prompts))):
        print(f"\n--- Example {i+1} ---")
        print("Proverb:", proverbs[i])
        print("Reference explanation:", references[i])
        print("Base output:", base_outputs[i])
        print("Tuned output:", tuned_outputs[i])

    # 8. Save results
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./fine_tuned_gpt2",
                        help="Directory of the fine-tuned model (HuggingFace format).")
    parser.add_argument("--base_model_name", type=str, default="gpt2",
                        help="Name or path of the base model (default: gpt2).")
    parser.add_argument("--data_path", type=str, default="./data/swahili_proverbs.csv",
                        help="Path to CSV/JSONL dataset. If missing, evaluate.py will try to import pairs from fine_tune.py.")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Max new tokens to generate per prompt.")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to evaluate (0 = all).")
    args = parser.parse_args()
    main(args)
