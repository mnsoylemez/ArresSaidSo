# ArresSaidSo
# QA Transformer: Question-Answering with Transformer Model

This repository contains a PyTorch-based Transformer Encoder-Decoder model for question-answering tasks, along with a preprocessing script to prepare a JSON dataset and train a SentencePiece BPE tokenizer. The model is trained on a dataset of question-answer pairs and can generate answers to user-provided questions.

## Features
- Transformer Encoder-Decoder architecture for question-answering.
- SentencePiece BPE tokenization for flexible text processing.
- Mixed precision training with gradient clipping for efficiency.
- Top-p sampling and repetition penalty for controlled answer generation.
- Early stopping and learning rate scheduling for robust training.
- Dataset validation and preprocessing for Q&A pairs.

## Prerequisites

### Hardware
- **RAM**: At least 16GB (32GB or higher recommended).
- **GPU**: Optional CUDA-compatible GPU for faster training (CPU supported but GPU is a must for big datasets/model size).
- **Storage**: ~1GB for dataset, model weights, and tokenizer files.

### Software
- **Python**: 3.8 or higher.
- **Operating System**: Linux, macOS, or Windows (Linux recommended).
- **Dependencies**:
  ```bash
  pip install torch>=2.0.0 sentencepiece>=0.1.99 tqdm>=4.66.1 numpy>=1.24.0
  ```
  Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Dataset
- A JSON file with question-answer pairs in the format:
  ```json
  [
      {"question": "What is project management?", "answer": "Project management is the process of leading a team to achieve specific goals within constraints like time, budget, and scope."},
      {"question": "What is AI?", "answer": "AI is the simulation of human intelligence in machines."},
      ...
  ]
  ```
- Recommended: At least 1,000 Q&A pairs for meaningful training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/qa-transformer.git
   cd qa-transformer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Place your JSON dataset in the `data/` directory (e.g., `data/testo.json`).
   - Alternatively, use the provided sample dataset or create your own.

## Usage

### Step 1: Preprocess the Dataset
Run the preprocessing script to validate the dataset and train the SentencePiece tokenizer:
```bash
python preprocess_data.py --input_json data/dataset.json --prepared_data data/prepared_data.txt --spm_prefix data/spm_prompt
```
- **Outputs**:
  - `data/prepared_data.txt`: Formatted Q&A text.
  - `data/spm_prompt.model`: SentencePiece model.
  - `data/spm_prompt.vocab`: Vocabulary file.
- Copy `data/spm_prompt.model` to `data/spm_bpe.model` for training:
  ```bash
  cp data/spm_prompt.model data/spm_bpe.model
  ```

### Step 2: Train the Model and Generate Answers
Run the training script to train the Transformer model and generate an example answer:
```bash
python qa_transformer.py --data_path data/dataset.json --sp_model_path data/spm_bpe.model --save_path models/qa_transformer.pth
```
- **Notes**:
  - Ensure `data/dataset.json` exists.
  - Training parameters can be adjusted in `qa_transformer.py` (see `Config` class).
  - The script saves the best model to `models/qa_transformer.pth` and generates an answer for “What is project management?”.

### Step 3: Generate Custom Answers
To generate answers for custom questions, modify `qa_transformer.py`:
```python
question = "Your custom question here"
prompt = f"Q: {question} A:"
generated_answer = generate(model, tokenizer, prompt, config)
print(f"Q: {question}")
print(f"A: {generated_answer}")
```
Then rerun:
```bash
python qa_transformer.py --data_path data/wikiqa_dataset.json --sp_model_path data/spm_bpe.model --save_path models/qa_transformer.pth
```

## Project Structure
```
qa-transformer/
├── data/
│   ├── dataset.json            # Dataset for training
│   ├── prepared_data.txt       # Preprocessed text
│   ├── spm_prompt.model        # SentencePiece model
│   ├── spm_prompt.vocab        # SentencePiece vocabulary
│   └── spm_bpe.model           # Copied SentencePiece model
├── models/
│   └── qa_transformer.pth      # Trained model weights
├── qa_transformer.py           # Main script for training/inference
├── preprocess_data.py          # Dataset preprocessing script
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── LICENSE                     # License file
└── .gitignore                  # Git ignore file
```

## Configuration
Key parameters in `qa_transformer.py` (edit `Config` class):
- `embed_dim`: Embedding size (default: 128).
- `num_heads`: Number of attention heads (default: 8).
- `num_layers`: Number of Transformer layers (default: 4).
- `block_size`: Maximum sequence length (default: 256).
- `learning_rate`: Initial learning rate (default: 2e-5).
- `batch_size`: Training batch size (default: 32).
- `max_iters`: Maximum training iterations (default: 10,000).

## Troubleshooting
- **File Not Found**: Ensure all input files (`dataset.json`, `spm_bpe.model`) exist in the `data/` directory.
- **CUDA Errors**: Set `device='cpu'` in `qa_transformer.py` (line: `self.device = "cpu"`) to run on CPU.
- **Poor Answer Quality**:
  - Increase dataset size (>1,000 Q&A pairs).
  - Adjust generation parameters (`temperature`, `top_p`, `repetition_penalty`).
  - Tune hyperparameters (`learning_rate`, `dropout`).
- **Training Divergence**: Increase `learning_rate` (e.g., to 1e-4) or reduce `dropout` (e.g., to 0.1).

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, open a GitHub issue or contact [soylemeznurhan@gmail.com](mailto:soylemeznurhan@gmail.com).
