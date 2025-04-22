import json
import sentencepiece as spm
import os
import random

# File paths
input_json_path = "/content/dataset.json"  # Input JSON file containing Q&A data
prepared_data_path = "/content/prepared_data.txt"  # File for preprocessed text
spm_model_prefix = "/content/spm_prompt"  # Prefix for the SentencePiece model files
PREDEFINED_VOCAB_SIZE = 10000  # Desired vocabulary size
NUM_SPECIAL_TOKENS = 4  # PAD, UNK, BOS, EOS

def validate_dataset(file_path):
    """
    Validates the dataset to ensure it meets format and content requirements.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Entry at index {i} is not a dictionary: {item}")
            question = item.get("question", None)
            answer = item.get("answer", None)
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(f"Invalid Q&A at index {i}. Must contain strings: {item}")
            if not question.strip() or not answer.strip():
                raise ValueError(f"Empty question or answer at index {i}: {item}")
        print("Dataset validation passed successfully.")
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        raise

# Step 1: Validate and preprocess dataset
try:
    print("Starting dataset validation and preprocessing...")
    validate_dataset(input_json_path)
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_lines = [
        f"Q: {item['question'].strip()} A: {item['answer'].strip()}"
        for item in data if "question" in item and "answer" in item
    ]
    print(f"Formatted {len(formatted_lines)} Q&A pairs.")
except Exception as e:
    print(f"Error during dataset preparation: {e}")
    raise

# Step 2: Shuffle and save preprocessed text
try:
    random.shuffle(formatted_lines)
    with open(prepared_data_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_lines))
    print(f"Preprocessed data saved to {prepared_data_path}.")
except Exception as e:
    print(f"Error while saving preprocessed data: {e}")
    raise

# Step 3: Train SentencePiece model using BPE
try:
    print("Starting SentencePiece model training with BPE...")
    spm.SentencePieceTrainer.train(
        input=prepared_data_path,
        model_prefix=spm_model_prefix,
        vocab_size=PREDEFINED_VOCAB_SIZE,
        model_type="bpe",  # Changed to BPE tokenization
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=1.0,  # Ensure full character coverage
        shuffle_input_sentence=True
    )
    # Check if model files were created
    model_file = spm_model_prefix + ".model"
    vocab_file = spm_model_prefix + ".vocab"
    if os.path.exists(model_file) and os.path.exists(vocab_file):
        print("SentencePiece model training complete.")
        print(f"Model file: {model_file}")
        print(f"Vocabulary file: {vocab_file}")
        print(f"Predefined Vocabulary Size Used: {PREDEFINED_VOCAB_SIZE}")
    else:
        raise FileNotFoundError("SentencePiece model files were not created.")
except Exception as e:
    print(f"Error during SentencePiece training: {e}")
    raise
