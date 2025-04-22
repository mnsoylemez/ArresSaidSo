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
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self):
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1."
        assert 0 <= self.train_ratio <= 1, "train_ratio must be between 0 and 1."
        assert 0 <= self.top_p <= 1, "top_p must be between 0 and 1."
        assert self.block_size > 0, "block_size must be a positive value"
        assert self.embed_dim > 0, "embed_dim must be a positive value"
        assert self.eval_iters > 0, "eval_iters must be a positive value"

class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size, step_size, prompt_format, device):
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
    
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        prompts = [self.prompt_format.format(question=item['question'], answer=item['answer']) for item in qa_pairs]
        text = ' '.join(prompts)
        tokens = self.tokenizer.encode(text, out_type=int)
        return tokens

class TransformerEncDec(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout, block_size, device, pad_token_id):
        super().__init__()
        self.embed_dim = embed_dim  # Add this line
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
    
    def forward(self, src, tgt=None):
        B, T = src.shape
        pos = torch.arange(0, T, dtype=torch.long, device=self.device).unsqueeze(0)
        src_emb = self.embed(src) + self.pos_emb(pos)
        
        src_mask = (src != self.pad_token_id)
        src_emb = src_emb * src_mask.unsqueeze(-1)
        
        memory = self.encoder(src_emb, src_key_padding_mask=~src_mask)
        
        if tgt is not None:
            tgt_seq_len = tgt.size(1)
            tgt_pos = torch.arange(0, tgt_seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            tgt_emb = self.embed(tgt) + self.pos_emb(tgt_pos)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=~src_mask)
        else:
            # During inference, you might want to handle tgt differently
            # For example, use the encoder's output to generate the target sequence
            # Here, we assume that tgt_emb should be initialized appropriately
            # This is a placeholder; adjust based on your specific use case
            tgt_emb = torch.zeros(B, 1, self.embed_dim, device=self.device)
            output = self.decoder(tgt_emb, memory, memory_key_padding_mask=~src_mask)
        
        logits = self.fc(output)
        loss = None
        if tgt is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=self.pad_token_id)
        return logits, loss

def create_dataloaders(config, tokenizer):
    dataset = QADataset(config.data_path, tokenizer, config.block_size, config.step_size, config.prompt_format, config.device)
    train_size = int(len(dataset) * config.train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model, val_loader, config):
    model.eval()
    losses = []
    for xb, yb in val_loader:
        xb, yb = xb.to(config.device), yb.to(config.device)
        logits, loss = model(xb, yb)
        losses.append(loss.item())
    val_loss = sum(losses) / len(losses) if losses else 0
    model.train()
    return val_loss

def create_model(config, tokenizer):
    vocab_size = tokenizer.get_piece_size()
    model = TransformerEncDec(vocab_size, config.embed_dim, config.num_heads, config.num_layers, config.dropout, config.block_size, config.device, config.pad_token_id).to(config.device)
    return model

def train_model(model, train_loader, val_loader, optimizer, scaler, scheduler, config):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    train_iter = iter(train_loader)
    pbar = tqdm(range(config.max_iters), desc="Training")
    for total_iters in pbar:
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        
        # Move tensors to the specified device
        xb, yb = xb.to(config.device), yb.to(config.device)
        
        optimizer.zero_grad()
        with autocast():
            logits, loss = model(xb, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix({'Train Loss': loss.item()})
        
        if total_iters % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, config)
            pbar.set_postfix({'Train Loss': loss.item(), 'Val Loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), config.save_path)
                print(f"\nModel saved with new best validation loss: {best_val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience_early_stopping:
                    print(f"\nStopping early due to lack of improvement for {config.patience_early_stopping} evaluations.")
                    return
            scheduler.step(val_loss)

@torch.no_grad()
def generate(model, tokenizer, prompt, config):
    model.eval()
    prompt_tokens = tokenizer.encode(prompt, out_type=int)
    if prompt_tokens[0] != tokenizer.bos_id():
        prompt_tokens = [tokenizer.bos_id()] + prompt_tokens
    buffer = torch.tensor([prompt_tokens], dtype=torch.long, device=config.device)
    
    generated_tokens = prompt_tokens.copy()
    
    while len(generated_tokens) < len(prompt_tokens) + config.block_size:
        if buffer.size(1) > config.block_size:
            buffer = buffer[:, -config.block_size:]
        src = buffer
        pos = torch.arange(src.size(1), dtype=torch.long, device=config.device).unsqueeze(0)
        logits, _ = model(src)
        logits = logits[:, -1, :] / config.temperature
        logits = logits.float()
        
        # Apply repetition penalty
        repetition_penalty = config.repetition_penalty
        for token_id in set(generated_tokens):
            if token_id < 0:
                continue
            logits[0, token_id] /= repetition_penalty
        
        # Apply top-p sampling
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > config.top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[:, indices_to_remove] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        idx_next = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(idx_next.item())
        buffer = torch.cat((buffer, idx_next), dim=1)
        
        if idx_next.item() == tokenizer.eos_id():
            break
    
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


if __name__ == "__main__":
    config = Config(step_size=128)
    config.validate()
    tokenizer = spm.SentencePieceProcessor(model_file=config.sp_model_path)
    config.pad_token_id = tokenizer.pad_id()
    config.bos_token_id = tokenizer.bos_id()
    config.eos_token_id = tokenizer.eos_id()

    if not os.path.exists(config.sp_model_path):
        print("Error: SentencePiece BPE model not found. Please provide the spm_bpe.model")
        exit()
    
    model = create_model(config, tokenizer)
    train_loader, val_loader = create_dataloaders(config, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.scheduler_patience, factor=config.scheduler_factor, verbose=config.scheduler_verbose)
    
    train_model(model, train_loader, val_loader, optimizer, scaler, scheduler, config)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    
    question = "What is project management?" #This would be the first question to your model. Replace the question according to your training data.
    prompt = f"Q: {question} A:"
    generated_answer = generate(model, tokenizer, prompt, config)
    print(f"Q: {question}")
    print(f"A: {generated_answer}")
