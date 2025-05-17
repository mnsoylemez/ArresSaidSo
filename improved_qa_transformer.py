import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import sentencepiece as spm
import os
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import math
from torch.utils.checkpoint import checkpoint

class Config:
    def __init__(self, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_dim = 128
        self.num_heads = 8
        self.num_layers = 4
        self.dropout = 0.3
        self.block_size = 256
        self.step_size = kwargs.get('step_size', self.block_size)
        self.batch_size = 32
        self.max_iters = 10000
        self.eval_interval = 1000
        self.learning_rate = 2e-5
        self.weight_decay = 1e-5
        self.save_path = "/content/qa_transformer.pth"
        self.sp_model_path = "/content/spm_bpe.model"
        self.data_path = "/content/dataset.json"
        self.train_ratio = 0.9
        self.temperature = 1.0
        self.top_k = 10
        self.top_p = 0.9
        self.repetition_penalty = 1.2
        self.gradient_clip = 0.5
        self.eval_iters = 1000
        self.scheduler_patience = 3
        self.scheduler_factor = 0.5
        self.scheduler_verbose = True
        self.patience_early_stopping = 5
        self.prompt_format = "Q: {question} A: {answer}"
        self.pad_token_id = None  # To be set later
        self.use_gradient_checkpointing = True  # Enable gradient checkpointing for memory efficiency
        self.max_generate_length = 100  # Maximum number of tokens to generate
        self.generation_batch_size = 4  # Batch size for generation
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self):
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1."
        assert 0 <= self.train_ratio <= 1, "train_ratio must be between 0 and 1."
        assert 0 <= self.top_p <= 1, "top_p must be between 0 and 1."
        assert self.block_size > 0, "block_size must be a positive value"
        assert self.embed_dim > 0, "embed_dim must be a positive value"
        assert self.eval_iters > 0, "eval_iters must be a positive value"
        assert self.max_generate_length > 0, "max_generate_length must be a positive value"

class QADataset(Dataset):
    """Dataset for question-answering tasks."""
    def __init__(self, file_path, tokenizer, block_size, step_size, prompt_format, device):
        """
        Initialize QA dataset.
        
        Args:
            file_path (str): Path to the JSON data file
            tokenizer: Tokenizer for encoding text
            block_size (int): Maximum sequence length
            step_size (int): Step size for sliding window
            prompt_format (str): Format string for Q&A pairs
            device: Device to store tensors on
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.step_size = step_size
        self.prompt_format = prompt_format
        self.device = device
        self.data = self.load_data(file_path)
    
    def __len__(self):
        if len(self.data) < self.block_size:
            return 1  # At least one batch with padding
        return (len(self.data) - self.block_size) // self.step_size + 1
    
    def __getitem__(self, idx):
        start = idx * self.step_size
        end = start + self.block_size
        x = self.data[start:end]
        y = self.data[start+1:end+1]
        
        # Pad sequences if necessary
        if len(x) < self.block_size:
            x = x + [self.tokenizer.pad_id()] * (self.block_size - len(x))
            y = y + [self.tokenizer.pad_id()] * (self.block_size - len(y))
        
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        return x, y
    
    def load_data(self, file_path):
        """Load and tokenize data from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            prompts = [self.prompt_format.format(question=item['question'], answer=item['answer']) for item in qa_pairs]
            text = ' '.join(prompts)
            tokens = self.tokenizer.encode(text, out_type=int)
            return tokens
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

class TransformerEncDec(nn.Module):
    """Transformer Encoder-Decoder model for question answering."""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout, block_size, device, pad_token_id):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size
        self.device = device
        self.pad_token_id = pad_token_id
        self.use_checkpointing = False  # Will be set in train method
    
    def _encoder_forward(self, src, src_mask):
        """Encoder forward pass with optional checkpointing."""
        if self.use_checkpointing and self.training:
            return checkpoint(lambda s, m: self.encoder(s, src_key_padding_mask=~m), src, src_mask)
        else:
            return self.encoder(src, src_key_padding_mask=~src_mask)
    
    def _decoder_forward(self, tgt_emb, memory, tgt_mask, src_mask):
        """Decoder forward pass with optional checkpointing."""
        if self.use_checkpointing and self.training:
            return checkpoint(
                lambda t, m, tm, sm: self.decoder(t, m, tgt_mask=tm, memory_key_padding_mask=~sm),
                tgt_emb, memory, tgt_mask, src_mask
            )
        else:
            return self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=~src_mask)
    
    def forward(self, src, tgt=None):
        """
        Forward pass for training and inference.
        
        Args:
            src: Input tensor of token ids [batch_size, seq_len]
            tgt: Target tensor of token ids or None during inference
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Loss value if tgt is provided, None otherwise
        """
        B, T = src.shape
        
        # Create position indices and get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=self.device).unsqueeze(0)
        src_emb = self.embed(src) + self.pos_emb(pos)
        
        # Create padding mask (1 for non-pad tokens, 0 for pad tokens)
        src_mask = (src != self.pad_token_id)
        
        # Run encoder
        memory = self._encoder_forward(src_emb, src_mask)
        
        # Conditional decoder execution
        if tgt is not None:
            # Training mode
            tgt_seq_len = tgt.size(1)
            tgt_pos = torch.arange(0, tgt_seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            tgt_emb = self.embed(tgt) + self.pos_emb(tgt_pos)
            
            # Create causal mask for transformer decoder
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
            
            # Run decoder
            output = self._decoder_forward(tgt_emb, memory, tgt_mask, src_mask)
            
            # Get logits and compute loss
            logits = self.fc(output)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=self.pad_token_id)
            return logits, loss
        else:
            # During inference, we return only the encoder output (memory)
            # The actual generation will be handled by the generate method
            return memory, None
    
    def encode(self, src):
        """Encode input sequence."""
        B, T = src.shape
        pos = torch.arange(0, T, dtype=torch.long, device=self.device).unsqueeze(0)
        src_emb = self.embed(src) + self.pos_emb(pos)
        src_mask = (src != self.pad_token_id)
        memory = self.encoder(src_emb, src_key_padding_mask=~src_mask)
        return memory, src_mask
    
    def decode_step(self, tgt, memory, src_mask):
        """Decode one step during generation."""
        tgt_seq_len = tgt.size(1)
        tgt_pos = torch.arange(0, tgt_seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(tgt.size(0), -1)
        tgt_emb = self.embed(tgt) + self.pos_emb(tgt_pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=~src_mask)
        logits = self.fc(output[:, -1:, :])  # Get logits only for the last position
        return logits

def create_dataloaders(config, tokenizer):
    """Create training and validation data loaders."""
    try:
        dataset = QADataset(config.data_path, tokenizer, config.block_size, config.step_size, config.prompt_format, config.device)
        
        # Create train-validation split
        train_size = int(len(dataset) * config.train_ratio)
        val_size = len(dataset) - train_size
        
        # Use a fixed random seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
        return train_loader, val_loader
    except Exception as e:
        raise RuntimeError(f"Error creating dataloaders: {str(e)}")

@torch.no_grad()
def evaluate(model, val_loader, config):
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    for xb, yb in val_loader:
        try:
            xb, yb = xb.to(config.device), yb.to(config.device)
            logits, loss = model(xb, yb)
            losses.append(loss.item())
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            continue
    val_loss = sum(losses) / len(losses) if losses else float('inf')
    model.train()
    return val_loss

def create_model(config, tokenizer):
    """Create the transformer model."""
    vocab_size = tokenizer.get_piece_size()
    model = TransformerEncDec(vocab_size, config.embed_dim, config.num_heads, config.num_layers, config.dropout, config.block_size, config.device, config.pad_token_id).to(config.device)
    return model

def train_model(model, train_loader, val_loader, optimizer, scaler, scheduler, config):
    """Train the model with mixed precision and gradient checkpointing."""
    model.train()
    model.use_checkpointing = config.use_gradient_checkpointing
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_iter = iter(train_loader)
    pbar = tqdm(range(config.max_iters), desc="Training")
    
    for total_iters in pbar:
        try:
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)
            
            optimizer.zero_grad()
            
            with autocast():
                logits, loss = model(xb, yb)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Apply gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({'Train Loss': f"{loss.item():.4f}"})
            
            # Evaluate periodically
            if total_iters % config.eval_interval == 0 or total_iters == config.max_iters - 1:
                val_loss = evaluate(model, val_loader, config)
                pbar.set_postfix({'Train Loss': f"{loss.item():.4f}", 'Val Loss': f"{val_loss:.4f}"})
                
                # Save checkpoint if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'config': config.__dict__,
                        'iter': total_iters,
                        'best_val_loss': best_val_loss
                    }
                    torch.save(checkpoint, config.save_path)
                    print(f"\nModel saved with new best validation loss: {best_val_loss:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience_early_stopping:
                        print(f"\nStopping early due to lack of improvement for {config.patience_early_stopping} evaluations.")
                        return
                
                # Update learning rate based on validation loss
                if scheduler:
                    scheduler.step(val_loss)
                
                # Switch back to training mode
                model.train()
        
        except Exception as e:
            print(f"Error in training iteration {total_iters}: {str(e)}")
            continue

@torch.no_grad()
def generate_batch(model, tokenizer, prompts, config):
    """
    Generate answers for a batch of questions.
    
    Args:
        model: The trained model
        tokenizer: SentencePiece tokenizer
        prompts: List of prompts to generate answers for
        config: Configuration object
        
    Returns:
        List of generated texts
    """
    model.eval()
    
    # Process prompts in batches
    batch_size = config.generation_batch_size
    all_results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize and pad all prompts in the batch
        encoded_prompts = []
        max_length = 0
        
        for prompt in batch_prompts:
            prompt_tokens = tokenizer.encode(prompt, out_type=int)
            if prompt_tokens[0] != tokenizer.bos_id():
                prompt_tokens = [tokenizer.bos_id()] + prompt_tokens
            encoded_prompts.append(prompt_tokens)
            max_length = max(max_length, len(prompt_tokens))
        
        # Pad to same length
        padded_prompts = []
        for prompt_tokens in encoded_prompts:
            padded = prompt_tokens + [tokenizer.pad_id()] * (max_length - len(prompt_tokens))
            padded_prompts.append(padded)
        
        # Convert to tensor
        src = torch.tensor(padded_prompts, dtype=torch.long, device=config.device)
        
        # Encode the source sequences
        memory, src_mask = model.encode(src)
        
        # Initialize target sequences with the first token (usually BOS)
        tgt = torch.tensor([[tokenizer.bos_id()] for _ in range(len(batch_prompts))], dtype=torch.long, device=config.device)
        
        # Store generated token ids for each sequence
        generated_sequences = [[] for _ in range(len(batch_prompts))]
        
        # Track which sequences have completed (generated EOS token)
        completed = [False] * len(batch_prompts)
        
        # Generate tokens sequentially
        for _ in range(config.max_generate_length):
            # Get logits for next token
            logits = model.decode_step(tgt, memory, src_mask)
            logits = logits[:, -1, :] / config.temperature
            
            # Apply repetition penalty
            for b in range(len(batch_prompts)):
                if not completed[b]:
                    for token_id in set(tgt[b].tolist()):
                        if token_id != tokenizer.pad_id():
                            logits[b, token_id] /= config.repetition_penalty
            
            # Apply top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Create a mask for tokens to remove
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0  # Keep the top-1 token
            
            # Apply the mask for each batch item
            for b in range(len(batch_prompts)):
                if not completed[b]:
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    probs[b, indices_to_remove] = 0
                    probs[b] = probs[b] / probs[b].sum()  # Renormalize
            
            # Sample next tokens
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_tokens], dim=1)
            
            # Store generated tokens and check for EOS
            for b in range(len(batch_prompts)):
                if not completed[b]:
                    token_id = next_tokens[b, 0].item()
                    generated_sequences[b].append(token_id)
                    if token_id == tokenizer.eos_id():
                        completed[b] = True
            
            # Break if all sequences have completed
            if all(completed):
                break
        
        # Decode the generated sequences and add to results
        for b, seq in enumerate(generated_sequences):
            # Trim if EOS token is present
            if tokenizer.eos_id() in seq:
                seq = seq[:seq.index(tokenizer.eos_id()) + 1]
            
            # Create full sequence including prompt
            full_sequence = encoded_prompts[b] + seq
            generated_text = tokenizer.decode(full_sequence)
            all_results.append(generated_text)
    
    return all_results

@torch.no_grad()
def generate(model, tokenizer, prompt, config):
    """Single prompt generation wrapper around batch generation."""
    return generate_batch(model, tokenizer, [prompt], config)[0]

def load_checkpoint(model, optimizer=None, scheduler=None, config=None):
    """Load model from checkpoint."""
    if os.path.exists(config.save_path):
        try:
            checkpoint = torch.load(config.save_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            print(f"Loaded checkpoint with validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
            return checkpoint.get('iter', 0), checkpoint.get('best_val_loss', float('inf'))
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return 0, float('inf')
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, float('inf')

if __name__ == "__main__":
    try:
        config = Config(step_size=128)
        config.validate()
        
        # Check if SentencePiece model exists
        if not os.path.exists(config.sp_model_path):
            print("Error: SentencePiece BPE model not found. Please provide the spm_bpe.model")
            exit(1)
        
        # Initialize tokenizer
        tokenizer = spm.SentencePieceProcessor(model_file=config.sp_model_path)
        config.pad_token_id = tokenizer.pad_id()
        config.bos_token_id = tokenizer.bos_id()
        config.eos_token_id = tokenizer.eos_id()
        
        # Create model
        model = create_model(config, tokenizer)
        
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = GradScaler()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.scheduler_patience, 
                                      factor=config.scheduler_factor, verbose=config.scheduler_verbose)
        
        # Try to load from checkpoint
        start_iter, best_val_loss = load_checkpoint(model, optimizer, scheduler, config)
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(config, tokenizer)
        
        # Train model
        train_model(model, train_loader, val_loader, optimizer, scaler, scheduler, config)
        
        # Load best model for generation
        model.load_state_dict(torch.load(config.save_path)['model_state_dict'])
        model.eval()
        
        # Example of batch generation
        questions = [
            "What is project management?",
            "How does artificial intelligence work?",
            "What are the benefits of renewable energy?"
        ]
        prompts = [f"Q: {q} A:" for q in questions]

        generated_answers = generate_batch(model, tokenizer, prompts, config)
        
        # Print results
        for q, a in zip(questions, generated_answers):
            print(f"\nQ: {q}")
            print(f"A: {a}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
