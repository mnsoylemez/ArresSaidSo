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
# import math # Removed as it's not directly used
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple, Optional, Any # Added for more specific type hints

class Config:
    """Configuration class for the QA Transformer model and training pipeline."""
    def __init__(self, **kwargs: Any): # Allow any keyword arguments for overrides
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu" # Device to use for computation ('cuda' or 'cpu')
        
        # Model architecture parameters
        self.embed_dim: int = 128            # Dimensionality of token embeddings and Transformer hidden states
        self.num_heads: int = 8              # Number of attention heads in MultiHeadAttention layers
        self.num_layers: int = 4             # Number of Transformer encoder/decoder layers
        self.dropout: float = 0.3            # Dropout rate used in Transformer layers and embeddings
        
        # Sequence processing parameters
        # `block_size` is the sequence length for QADataset processing (input chunk size for creating x, y pairs).
        self.block_size: int = 256           
        # `max_positional_embeddings` defines the size of the positional embedding table in TransformerEncDec.
        # This is the ultimate sequence length limit the model can handle directly with its learned PEs.
        self.max_positional_embeddings: int = 2048 
        
        # Dataset and DataLoader parameters
        self.step_size: int = kwargs.get('step_size', self.block_size) # Step size for QADataset sliding window
        self.batch_size: int = 32            # Batch size for training
        self.train_ratio: float = 0.9        # Proportion of data to use for training (rest is for validation)
        self.prompt_format: str = "Q: {question} A: {answer}" # Format string for Q&A pairs in the dataset

        # Training parameters
        self.max_iters: int = 10000          # Maximum number of training iterations
        self.eval_interval: int = 1000       # Interval (in iterations) for performing evaluation on the validation set
        self.learning_rate: float = 2e-5     # Initial learning rate for the AdamW optimizer
        self.weight_decay: float = 1e-5      # Weight decay for the AdamW optimizer
        self.gradient_clip: float = 0.5      # Value for gradient clipping to prevent exploding gradients
        self.use_gradient_checkpointing: bool = True # Whether to use gradient checkpointing for memory efficiency
        
        # Scheduler parameters (for ReduceLROnPlateau)
        self.scheduler_patience: int = 3     # Number of evaluations with no improvement after which learning rate will be reduced
        self.scheduler_factor: float = 0.5   # Factor by which the learning rate will be reduced (new_lr = lr * factor)
        self.scheduler_verbose: bool = True  # If True, prints a message when the learning rate is updated
        
        # Early stopping parameters
        self.patience_early_stopping: int = 5 # Number of evaluations with no improvement after which training will be stopped

        # File paths - can be overridden by kwargs.
        self.save_path: str = "./qa_transformer.pth"    # Path to save/load model checkpoint
        self.sp_model_path: str = "./spm_bpe.model" # Path to SentencePiece tokenizer model
        self.data_path: str = "./dataset.json"      # Path to JSON dataset file
        
        # Generation parameters (for `generate_batch` function)
        self.temperature: float = 1.0        # Temperature for sampling (controls randomness; >1 more random, <1 less random)
        self.top_k: int = 10                 # Top-k sampling parameter (currently not used if top_p is active in `generate_batch`)
        self.top_p: float = 0.9              # Top-p (nucleus) sampling parameter (filters vocabulary to tokens with cumulative prob > top_p)
        self.repetition_penalty: float = 1.2 # Penalty for repeating tokens during generation
        self.max_generate_length: int = 100  # Maximum number of new tokens to generate in a sequence
        self.generation_batch_size: int = 4  # Batch size for the `generate_batch` function
        
        # Tokenizer-related IDs - these are typically set after tokenizer initialization
        self.pad_token_id: Optional[int] = None # Using Optional as it's None initially
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        
        # `eval_iters` is defined but not directly used to limit batches in the current `evaluate` function,
        # as `evaluate` iterates through the entire `val_loader`.
        # It's kept for potential future use where evaluation might be limited to a fixed number of batches.
        self.eval_iters: int = 1000 

        # Apply any overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}' provided.")
    
    def validate(self) -> None:
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1."
        assert 0 <= self.train_ratio <= 1, "train_ratio must be between 0 and 1."
        assert 0 <= self.top_p <= 1, "top_p must be between 0 and 1."
        assert self.block_size > 0, "block_size must be a positive value"
        assert self.embed_dim > 0, "embed_dim must be a positive value"
        # self.eval_iters is now an active attribute, so this assertion is fine.
        assert self.eval_iters > 0, "eval_iters must be a positive value" 
        assert self.max_generate_length > 0, "max_generate_length must be a positive value"

class QADataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]): # Hinting for Dataset item type
    """Dataset for question-answering tasks."""
    def __init__(self, 
                 file_path: str, 
                 tokenizer: spm.SentencePieceProcessor, 
                 block_size: int, 
                 step_size: int, 
                 prompt_format: str, 
                 device: str):
        """
        Initialize QA dataset.
        
        Args:
            file_path (str): Path to the JSON data file.
            tokenizer (spm.SentencePieceProcessor): Tokenizer for encoding text.
            block_size (int): Maximum sequence length for model input (x and y).
            step_size (int): Step size for creating overlapping chunks from the dataset.
            prompt_format (str): Format string for combining questions and answers.
            device (str): Device to store tensors on (e.g., "cpu", "cuda").
        """
        self.tokenizer: spm.SentencePieceProcessor = tokenizer
        self.block_size: int = block_size
        self.step_size: int = step_size
        self.prompt_format: str = prompt_format
        self.device: str = device
        self.data: List[int] = self.load_data(file_path) # Tokenized data
    
    def __len__(self) -> int:
        if not self.data:
            return 0
        # Number of possible starting positions for a chunk
        return (len(self.data) - 1) // self.step_size + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_offset = idx * self.step_size
        # We need block_size tokens for x and block_size for y, with y shifted by 1.
        # So, we attempt to grab block_size + 1 tokens from the data.
        end_offset = start_offset + self.block_size + 1
        
        chunk = self.data[start_offset:end_offset]
        
        x_tokens = chunk[:-1]
        y_tokens = chunk[1:]
        
        # Pad x_tokens to self.block_size
        x_padding_needed = self.block_size - len(x_tokens)
        if x_padding_needed > 0:
            x_tokens = x_tokens + [self.tokenizer.pad_id()] * x_padding_needed

        # Pad y_tokens to self.block_size
        y_padding_needed = self.block_size - len(y_tokens)
        if y_padding_needed > 0:
            y_tokens = y_tokens + [self.tokenizer.pad_id()] * y_padding_needed
            
        x = torch.tensor(x_tokens, dtype=torch.long, device=self.device)
        y = torch.tensor(y_tokens, dtype=torch.long, device=self.device)
        return x, y
    
    def load_data(self, file_path: str) -> List[int]:
        """Load and tokenize data from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            prompts = []
            for i, item in enumerate(qa_pairs):
                try:
                    prompts.append(self.prompt_format.format(question=item['question'], answer=item['answer']))
                except KeyError as e:
                    raise ValueError(f"Missing key {str(e)} in item {i} in file: {file_path}")
            
            text = ' '.join(prompts)
            tokens = self.tokenizer.encode(text, out_type=int)
            return tokens
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

class TransformerEncDec(nn.Module):
    """Transformer Encoder-Decoder model for question answering."""
    def __init__(self, 
                 vocab_size: int, 
                 embed_dim: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dropout: float, 
                 max_seq_len: int,  # Max sequence length for positional embeddings and processing
                 device: str, 
                 pad_token_id: int, 
                 use_checkpointing: bool = False):
        super().__init__()
        self.embed_dim: int = embed_dim
        self.embed: nn.Embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_emb: nn.Embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Standard Transformer encoder layer
        encoder_layer_args = {'d_model': embed_dim, 'nhead': num_heads, 'dropout': dropout, 'batch_first': True}
        _encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(**encoder_layer_args)
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(_encoder_layer, num_layers)
        
        # Standard Transformer decoder layer
        decoder_layer_args = {'d_model': embed_dim, 'nhead': num_heads, 'dropout': dropout, 'batch_first': True}
        _decoder_layer: nn.TransformerDecoderLayer = nn.TransformerDecoderLayer(**decoder_layer_args)
        self.decoder: nn.TransformerDecoder = nn.TransformerDecoder(_decoder_layer, num_layers)
        
        self.fc: nn.Linear = nn.Linear(embed_dim, vocab_size) # Final fully connected layer to map to vocabulary
        
        self.max_seq_len: int = max_seq_len # Maximum sequence length the model can process
        self.device: str = device
        self.pad_token_id: int = pad_token_id # ID of the padding token
        self.use_checkpointing: bool = use_checkpointing # Whether to use gradient checkpointing

    def _run_encoder(self, src_emb: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper to run encoder, facilitating checkpointing."""
        return self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def _run_decoder(self, tgt_emb: torch.Tensor, memory: torch.Tensor, 
                       tgt_mask: Optional[torch.Tensor] = None, 
                       memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper to run decoder, facilitating checkpointing."""
        return self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)

    def _encoder_forward(self, src_emb: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Encoder forward pass with optional checkpointing.
        `src_key_padding_mask` should be True for pad tokens, False otherwise.
        """
        if self.use_checkpointing and self.training:
            return checkpoint(self._run_encoder, src_emb, src_key_padding_mask, use_reentrant=False)
        else:
            return self._run_encoder(src_emb, src_key_padding_mask)
    
    def _decoder_forward(self, 
                         tgt_emb: torch.Tensor, 
                         memory: torch.Tensor, 
                         tgt_causal_mask: torch.Tensor, # Causal mask for target self-attention
                         memory_key_padding_mask: torch.Tensor # Padding mask for memory (encoder output)
                        ) -> torch.Tensor:
        """
        Decoder forward pass with optional checkpointing.
        `memory_key_padding_mask` should be True for pad tokens in memory, False otherwise.
        `tgt_causal_mask` is the causal mask.
        """
        if self.use_checkpointing and self.training:
            return checkpoint(self._run_decoder, tgt_emb, memory, tgt_causal_mask, memory_key_padding_mask, use_reentrant=False)
        else:
            return self._run_decoder(tgt_emb, memory, tgt_causal_mask, memory_key_padding_mask)
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training and inference.
        
        Args:
            src: Input tensor of token ids [batch_size, seq_len]
            tgt: Target tensor of token ids or None during inference
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Loss value if tgt is provided, None otherwise
        """
        B, T_orig = src.shape
        
        # Truncate src if its length exceeds max_seq_len
        if T_orig > self.max_seq_len:
            src = src[:, :self.max_seq_len]
            T = self.max_seq_len
        else:
            T = T_orig
            
        # Create position indices and get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=self.device).unsqueeze(0)
        src_emb = self.embed(src) + self.pos_emb(pos)
        
        # Create padding mask for source (True for pad tokens, False for non-pad tokens)
        src_key_padding_mask = (src == self.pad_token_id) 
        
        # Run encoder
        memory = self._encoder_forward(src_emb, src_key_padding_mask)
        
        # Conditional decoder execution (typically for training)
        if tgt is not None:
            # Process target sequence for decoder input
            T_tgt_orig = tgt.shape[1]
            # Truncate target sequence if longer than max_seq_len
            if T_tgt_orig > self.max_seq_len:
                tgt_proc = tgt[:, :self.max_seq_len] # Use truncated target
                tgt_proc_seq_len = self.max_seq_len
            else:
                tgt_proc = tgt
                tgt_proc_seq_len = T_tgt_orig

            tgt_pos_indices = torch.arange(0, tgt_proc_seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            tgt_emb = self.embed(tgt_proc) + self.pos_emb(tgt_pos_indices)
            
            # Create causal mask for target self-attention (prevents attending to future tokens)
            tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_proc_seq_len, device=self.device)
            
            # Decoder's memory_key_padding_mask is the padding mask from the source.
            output = self._decoder_forward(tgt_emb, memory, tgt_causal_mask, src_key_padding_mask)
            
            # Get logits and compute loss
            logits = self.fc(output)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_proc.view(-1), ignore_index=self.pad_token_id)
            return logits, loss
        else:
            # During inference (e.g. when only `encode` is called, then `decode_step` repeatedly)
            # This path of forward() is typically not used for full generation,
            # but `encode` provides the `memory`.
            return memory, None # Return memory and no loss
    
    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes an input sequence.
        Used during training (via forward) and for setting up inference.

        Args:
            src (torch.Tensor): Input token IDs (batch_size, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - memory (torch.Tensor): Encoder output (batch_size, seq_len, embed_dim).
                - src_key_padding_mask (torch.Tensor): Padding mask for `memory` (batch_size, seq_len), True for padded tokens.
        """
        B, T_orig = src.shape
        # Truncate src if its length exceeds model's capacity
        if T_orig > self.max_seq_len:
            src = src[:, :self.max_seq_len]
            T = self.max_seq_len
        else:
            T = T_orig
            
        pos_indices = torch.arange(0, T, dtype=torch.long, device=self.device).unsqueeze(0)
        src_emb = self.embed(src) + self.pos_emb(pos_indices)
        
        src_key_padding_mask = (src == self.pad_token_id) # True for pad tokens
        
        # Use _encoder_forward to potentially leverage checkpointing if encode is called during training.
        # However, typically self.training would be false if encode is called standalone for inference.
        memory = self._encoder_forward(src_emb, src_key_padding_mask)
        return memory, src_key_padding_mask
    
    def decode_step(self, tgt_tokens: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Decode one step during generation."""
        B, current_tgt_len = tgt.shape

        # If current target length exceeds PE table capacity, use a sliding window approach
        """
        Performs a single decoding step. Used for autoregressive generation.

        Args:
            tgt_tokens (torch.Tensor): Current sequence of generated tokens (batch_size, current_seq_len).
            memory (torch.Tensor): Encoder output (batch_size, src_seq_len, embed_dim).
            memory_key_padding_mask (Optional[torch.Tensor]): Padding mask for `memory`. True for padded tokens.

        Returns:
            torch.Tensor: Logits for the next token (batch_size, 1, vocab_size).
        """
        B, current_tgt_len = tgt_tokens.shape

        # If current target length exceeds PE table capacity, use a sliding window approach
        if current_tgt_len > self.max_seq_len:
            tgt_effective = tgt_tokens[:, -self.max_seq_len:] # Keep the most recent self.max_seq_len tokens
            effective_len = self.max_seq_len
        else:
            tgt_effective = tgt_tokens
            effective_len = current_tgt_len
        
        pos_indices = torch.arange(0, effective_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, -1)
        tgt_emb = self.embed(tgt_effective) + self.pos_emb(pos_indices)
        
        # Causal mask for target self-attention
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(effective_len, device=self.device)
        
        # Use _decoder_forward to potentially leverage checkpointing if decode_step is somehow used in training.
        # Typically self.training would be false here.
        output = self._decoder_forward(tgt_emb, memory, tgt_causal_mask, memory_key_padding_mask)
        
        # Get logits only for the last position of the output sequence
        logits = self.fc(output[:, -1:, :])
        return logits

def create_dataloaders(config: Config, tokenizer: spm.SentencePieceProcessor) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation data loaders from the QADataset.

    Args:
        config (Config): Configuration object.
        tokenizer (spm.SentencePieceProcessor): Initialized SentencePiece tokenizer.

    Returns:
        Tuple[DataLoader, DataLoader]: Training DataLoader and Validation DataLoader.
        
    Raises:
        RuntimeError: If there's an error during dataset instantiation or DataLoader creation.
    """
    try:
        dataset = QADataset(
            file_path=config.data_path, 
            tokenizer=tokenizer, 
            block_size=config.block_size, 
            step_size=config.step_size, 
            prompt_format=config.prompt_format, 
            device=config.device
        )
        
        if not dataset: # Or len(dataset) == 0, though QADataset.__len__ handles empty data.
             raise ValueError("Dataset is empty. Check data_path and dataset loading.")

        # Create train-validation split
        # Ensure val_size is not negative if dataset is very small
        train_len = int(len(dataset) * config.train_ratio)
        val_len = len(dataset) - train_len
        
        if train_len <= 0 or val_len <= 0:
            # This can happen if the dataset is too small for the given train_ratio.
            # Decide on a behavior: error out, or use the whole dataset for training, etc.
            # For now, let random_split handle it or error if it can't.
            # PyTorch random_split will raise error if lengths are 0 or sum doesn't match.
            if len(dataset) > 0 and (train_len == 0 or val_len == 0):
                 print(f"Warning: Dataset size {len(dataset)} is very small. Training with {train_len} samples, validating with {val_len} samples. Consider a larger dataset or different train_ratio.")


        # Use a fixed random seed for reproducible splits
        generator = torch.Generator().manual_seed(42) 
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0) # Typically no shuffle for val_loader
        return train_loader, val_loader
    except Exception as e: # Catching specific errors like ValueError from random_split is also an option
        # Log the original exception for more detailed debugging if necessary
        # import traceback; traceback.print_exc() 
        raise RuntimeError(f"Error creating dataloaders: {str(e)}")

@torch.no_grad()
def evaluate(model: TransformerEncDec, val_loader: DataLoader, config: Config) -> float:
    """
    Evaluates the model on the validation set.

    Args:
        model (TransformerEncDec): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation set.
        config (Config): Configuration object.

    Returns:
        float: The average validation loss. Returns float('inf') if validation fails or no data.
    """
    model.eval() # Set model to evaluation mode
    losses = []
    for xb, yb in val_loader:
        try:
            xb, yb = xb.to(config.device), yb.to(config.device)
            logits, loss = model(xb, yb)
            losses.append(loss.item())
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            continue
    val_loss = sum(losses) / len(losses) if losses else float('inf') # Calculate average loss, handle empty losses
    model.train() # Set model back to training mode
    return val_loss

def create_model(config: Config, tokenizer: spm.SentencePieceProcessor) -> TransformerEncDec:
    """
    Creates and initializes the TransformerEncDec model.

    Args:
        config (Config): Configuration object.
        tokenizer (spm.SentencePieceProcessor): Initialized SentencePiece tokenizer.
                                                 Used to get vocab_size and potentially special token IDs.
    Returns:
        TransformerEncDec: The initialized Transformer model.
    """
    vocab_size = tokenizer.get_piece_size() 
    if config.pad_token_id is None: # Should be set by main after tokenizer load
        raise ValueError("config.pad_token_id is not set. Initialize tokenizer and set it in Config first.")

    model = TransformerEncDec(
        vocab_size=vocab_size, 
        embed_dim=config.embed_dim, 
        num_heads=config.num_heads, 
        num_layers=config.num_layers, 
        dropout=config.dropout, 
        max_seq_len=config.max_positional_embeddings,
        device=config.device, 
        pad_token_id=config.pad_token_id, # Ensure this is correctly passed
        use_checkpointing=config.use_gradient_checkpointing
    ).to(config.device)
    return model

def train_model(model: TransformerEncDec, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                scaler: GradScaler, 
                scheduler: Optional[ReduceLROnPlateau], 
                config: Config) -> None:
    """
    Trains the TransformerEncDec model.

    Args:
        model (TransformerEncDec): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer.
        scaler (GradScaler): Gradient scaler for mixed-precision training.
        scheduler (Optional[ReduceLROnPlateau]): Learning rate scheduler.
        config (Config): Configuration object.
    """
    model.train() # Ensure model is in training mode
    
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
                    # Ensure the directory for the save_path exists
                    if config.save_path:
                        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
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
    
    # Process prompts in batches according to `config.generation_batch_size`
    batch_size = config.generation_batch_size
    all_results = []
    
    for i in range(0, len(prompts), batch_size):
        current_batch_prompts = prompts[i:i+batch_size]
        num_prompts_in_batch = len(current_batch_prompts)

        # --- 1. Tokenize and Prepare Input Prompts ---
        encoded_prompts_list = [] # Stores tokenized prompts before padding
        max_prompt_len = 0 # Max tokenized length in this specific batch of prompts
        
        for prompt_text in current_batch_prompts:
            # Tokenize the text prompt. Assumes SentencePiece tokenizer.encode does not add BOS/EOS by default.
            prompt_tokens = tokenizer.encode(prompt_text, out_type=int)
            
            # Prepend BOS token if not already present.
            # This is important if the model was trained to expect sequences to start with BOS.
            if not prompt_tokens or prompt_tokens[0] != tokenizer.bos_id():
                prompt_tokens = [tokenizer.bos_id()] + prompt_tokens
            
            encoded_prompts_list.append(prompt_tokens)
            max_prompt_len = max(max_prompt_len, len(prompt_tokens))
        
        # Pad all tokenized prompts in the current batch to the same length (max_prompt_len)
        # This is necessary for creating a tensor for batch processing.
        padded_prompts_tensor_list = []
        for prompt_tokens in encoded_prompts_list:
            padding_needed = max_prompt_len - len(prompt_tokens)
            padded_tokens = prompt_tokens + [tokenizer.pad_id()] * padding_needed
            padded_prompts_tensor_list.append(padded_tokens)
        
        # Convert list of padded prompts to a PyTorch tensor
        # src shape: [num_prompts_in_batch, max_prompt_len]
        src = torch.tensor(padded_prompts_tensor_list, dtype=torch.long, device=config.device)
        
        # --- 2. Encode Prompts ---
        # Obtain encoded representation (memory) and padding mask from the model's encoder.
        # `src_mask` is True for non-pad tokens, False for pad tokens.
        # The model's internal `memory_key_padding_mask` will use `~src_mask`.
        memory, src_mask = model.encode(src) # src is already truncated to model.max_seq_len inside encode()
        
        # --- 3. Initialize Generation Targets ---
        # Start generation for each prompt with the BOS token.
        # tgt shape: [num_prompts_in_batch, 1]
        tgt = torch.tensor([[tokenizer.bos_id()] for _ in range(num_prompts_in_batch)], dtype=torch.long, device=config.device)
        
        # List to store lists of generated token IDs for each prompt in the batch
        generated_sequences = [[] for _ in range(num_prompts_in_batch)]
        
        # Boolean flags to track if EOS has been generated for each sequence
        completed = [False] * num_prompts_in_batch # Corrected: use num_prompts_in_batch
        
        # --- 4. Autoregressive Generation Loop ---
        for _ in range(config.max_generate_length):
            # Get logits for the next token from the model's decoder.
            # `model.decode_step` handles truncation of `tgt` if it exceeds `model.max_seq_len`.
            # `src_mask` (from encoder) is used for `memory_key_padding_mask`.
            # Causal masking for `tgt` is handled within `decode_step`.
            logits_all_steps = model.decode_step(tgt, memory, src_mask) # Shape: [batch, current_tgt_len, vocab_size]
            
            # We only need logits for the *last* token in the `tgt` sequence to predict the *next* token.
            # Shape: [batch, vocab_size]
            logits_next_token = logits_all_steps[:, -1, :] 
            
            # Apply temperature scaling to logits.
            # Temperature < 1.0 makes distribution sharper (less random), > 1.0 makes it flatter (more random).
            logits_next_token = logits_next_token / config.temperature
            
            # Apply repetition penalty.
            # This discourages the model from repeating tokens that are already in the current `tgt` sequence.
            for batch_idx in range(num_prompts_in_batch):
                if not completed[batch_idx]: # Only apply to sequences that are still generating
                    # Iterate over unique tokens in the current target sequence for this batch item.
                    # Using `set` avoids penalizing multiple occurrences of the same token more than once.
                    for token_id_in_tgt in set(tgt[batch_idx].tolist()):
                        # Do not penalize padding token. Consider if other special tokens (BOS, EOS) should be excluded.
                        # Current implementation only excludes PAD.
                        if token_id_in_tgt != tokenizer.pad_id():
                            logits_next_token[batch_idx, token_id_in_tgt] /= config.repetition_penalty
            
            # Apply top-p (nucleus) sampling.
            # This filters the vocabulary to the smallest set of tokens whose cumulative probability exceeds `config.top_p`.
            probs = F.softmax(logits_next_token, dim=-1) # Convert logits to probabilities
            
            # Sort probabilities in descending order to identify tokens with highest likelihood.
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            
            # Calculate cumulative probabilities.
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # Shape: [batch, vocab_size]
            
            # Create a mask for tokens to remove (those that are not part of the nucleus).
            # Tokens are removed if their cumulative probability *exceeds* top_p.
            # The first token (highest prob) is always kept.
            sorted_indices_to_remove = cumulative_probs > config.top_p
            # Shift the mask: if prob[i] made cumulative_prob > top_p, then token i is kept, but token i+1 and onwards (in sorted order) are removed.
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0 # Ensure the token with the highest probability is never removed.
            
            # Apply the mask to the original probabilities tensor (probs).
            # For each batch item, gather the indices of tokens to be zeroed out.
            for batch_idx in range(num_prompts_in_batch):
                if not completed[batch_idx]: # Only process active sequences
                    # Get the actual token indices corresponding to `sorted_indices_to_remove`.
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    probs[batch_idx, indices_to_remove] = 0 # Zero out probabilities of tokens outside the nucleus
                    
                    # Renormalize the probabilities for the remaining tokens to sum to 1.
                    # This is crucial for `torch.multinomial`.
                    current_probs_sum = torch.sum(probs[batch_idx])
                    if current_probs_sum > 1e-9: # Check for sum being close to zero
                        probs[batch_idx] = probs[batch_idx] / current_probs_sum
                    else:
                        # If all probabilities become zero (e.g. due to extreme top_p or penalties),
                        # fall back to a uniform distribution over non-pad tokens to avoid errors.
                        # This is a recovery mechanism, ideally should not be hit frequently.
                        num_vocab = probs.shape[-1]
                        fallback_probs = torch.ones(num_vocab, device=config.device)
                        if tokenizer.pad_id() >= 0 and tokenizer.pad_id() < num_vocab: # Check pad_id is valid
                             fallback_probs[tokenizer.pad_id()] = 0 
                        if fallback_probs.sum() > 1e-9:
                           probs[batch_idx] = fallback_probs / fallback_probs.sum()
                        # If fallback_probs sum is also zero (e.g. vocab size 1 and it's pad_id), multinomial will error.
                        # This is an extreme edge case. For now, we let it be. A small epsilon could be added.

            # Sample the next token from the modified probability distribution.
            next_tokens = torch.multinomial(probs, num_samples=1) # Shape: [batch, 1]
            
            # Append the newly generated tokens to the `tgt` sequence.
            tgt = torch.cat([tgt, next_tokens], dim=1)
            
            # Store the generated token and check for EOS for each sequence in the batch.
            for batch_idx in range(num_prompts_in_batch): # Corrected: use num_prompts_in_batch
                if not completed[batch_idx]:
                    token_id = next_tokens[batch_idx, 0].item()
                    generated_sequences[batch_idx].append(token_id)
                    if token_id == tokenizer.eos_id():
                        completed[batch_idx] = True
            
            # Break the generation loop if all sequences in the batch have completed (generated EOS).
            if all(completed):
                break
        
        # --- 5. Decode Generated Sequences and Finalize Results ---
        for batch_idx, generated_token_ids in enumerate(generated_sequences): # Corrected: use batch_idx and generated_token_ids
            # Trim sequence at the first EOS token if present. Includes the EOS token itself.
            try:
                eos_index = generated_token_ids.index(tokenizer.eos_id())
                final_token_ids = generated_token_ids[:eos_index + 1]
            except ValueError:
                # EOS token not found; use the sequence as is (it was generated up to max_generate_length).
                final_token_ids = generated_token_ids
            
            # Concatenate the original tokenized prompt (which includes BOS) with the generated token IDs.
            # `encoded_prompts_list` stores the tokenized versions of `current_batch_prompts`.
            full_sequence_tokens = encoded_prompts_list[batch_idx] + final_token_ids
            
            # Decode the full token sequence back to text.
            generated_text = tokenizer.decode(full_sequence_tokens)
            all_results.append(generated_text)
    
    return all_results

@torch.no_grad()
def generate(model: TransformerEncDec, 
             tokenizer: spm.SentencePieceProcessor, 
             prompt: str, 
             config: Config) -> str:
    """
    Generates a single text sequence from a prompt using batch generation with batch size 1.

    Args:
        model (TransformerEncDec): The trained model.
        tokenizer (spm.SentencePieceProcessor): The tokenizer.
        prompt (str): The input prompt string.
        config (Config): Configuration object.

    Returns:
        str: The generated text.
    """
    return generate_batch(model, tokenizer, [prompt], config)[0]

def load_checkpoint(model: TransformerEncDec, 
                    optimizer: Optional[torch.optim.Optimizer] = None, 
                    scheduler: Optional[ReduceLROnPlateau] = None, 
                    config: Config = None) -> Tuple[int, float]:
    """
    Loads model, optimizer, and scheduler states from a checkpoint file.

    Args:
        model (TransformerEncDec): The model to load state into.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to load state into.
        scheduler (Optional[ReduceLROnPlateau]): The scheduler to load state into.
        config (Config): Configuration object, used for `save_path` and `device`.

    Returns:
        Tuple[int, float]:
            - start_iter (int): The iteration number to resume training from.
            - best_val_loss (float): The best validation loss achieved so far.
    """
    if config is None:
        # This case should ideally not be hit if called from main, but good for standalone use.
        print("Warning: Config object not provided to load_checkpoint. Using default save_path and device.")
        config = Config() # Fallback to default config, might not be what user wants.

    if not config.save_path or not os.path.exists(config.save_path):
        print(f"No checkpoint found at {config.save_path if config.save_path else 'default path'}. Starting from scratch.")
        return 0, float('inf')
        
    try:
        checkpoint_data = torch.load(config.save_path, map_location=config.device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data and checkpoint_data['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        loaded_iter = checkpoint_data.get('iter', 0)
        loaded_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from {config.save_path} (iteration {loaded_iter}, validation loss: {loaded_val_loss:.4f})")
        return loaded_iter, loaded_val_loss
    except Exception as e:
        print(f"Error loading checkpoint from {config.save_path}: {str(e)}. Starting from scratch.")
        return 0, float('inf')
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, float('inf')

if __name__ == "__main__":
    try:
        # --- Configuration Setup ---
        # Define paths for model, tokenizer, and data.
        # Users should modify these paths to match their environment.
        # Example:
        # model_checkpoint_path = "/path/to/your/models/qa_transformer.pth"
        # tokenizer_model_path = "/path/to/your/tokenizer/spm_bpe.model"
        # training_data_path = "/path/to/your/data/dataset.json"

        # Using default relative paths for this example:
        model_checkpoint_path = "qa_transformer.pth" 
        tokenizer_model_path = "spm_bpe.model"
        training_data_path = "dataset.json"

        # Initialize configuration, potentially overriding defaults with command-line args or other sources.
        # For this script, we directly pass them or rely on Config defaults.
        config = Config(
            step_size=128, # Example of overriding a non-path parameter
            save_path=model_checkpoint_path,
            sp_model_path=tokenizer_model_path,
            data_path=training_data_path
            # Add other overrides as needed, e.g.:
            # batch_size=64,
            # learning_rate=1e-4
        )
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
