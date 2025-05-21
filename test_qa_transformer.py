import unittest
from unittest.mock import patch, MagicMock, mock_open
import torch
import improved_qa_transformer as iqt # Assuming your file is improved_qa_transformer.py
import sentencepiece as spm # Required for type hinting if not already imported in iqt for QADataset tokenizer type

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        config = iqt.Config()
        self.assertEqual(config.embed_dim, 128)
        self.assertEqual(config.block_size, 256)
        self.assertEqual(config.save_path, "./qa_transformer.pth") # Check updated default
        self.assertTrue(config.use_gradient_checkpointing)
        self.assertEqual(config.max_positional_embeddings, 2048)

    def test_override(self):
        config = iqt.Config(embed_dim=256, block_size=512, save_path="/test/path.pth")
        self.assertEqual(config.embed_dim, 256)
        self.assertEqual(config.block_size, 512)
        self.assertEqual(config.save_path, "/test/path.pth")

    def test_unknown_kwargs(self):
        # This test relies on the print warning, might be better to capture stdout or log
        # For now, just ensure it doesn't crash
        try:
            _ = iqt.Config(unknown_param=123)
        except Exception as e:
            self.fail(f"Config raised an exception with unknown kwargs: {e}")


    def test_validation_dropout(self):
        with self.assertRaisesRegex(AssertionError, "Dropout must be between 0 and 1."):
            iqt.Config(dropout=-0.1).validate()
        with self.assertRaisesRegex(AssertionError, "Dropout must be between 0 and 1."):
            iqt.Config(dropout=1.1).validate()
        try:
            iqt.Config(dropout=0.5).validate() # Should not raise
        except AssertionError:
            self.fail("validate() raised AssertionError unexpectedly for valid dropout.")

    def test_validation_train_ratio(self):
        with self.assertRaisesRegex(AssertionError, "train_ratio must be between 0 and 1."):
            iqt.Config(train_ratio=-0.1).validate()
        with self.assertRaisesRegex(AssertionError, "train_ratio must be between 0 and 1."):
            iqt.Config(train_ratio=1.1).validate()

    def test_validation_top_p(self):
        with self.assertRaisesRegex(AssertionError, "top_p must be between 0 and 1."):
            iqt.Config(top_p=-0.1).validate()
        with self.assertRaisesRegex(AssertionError, "top_p must be between 0 and 1."):
            iqt.Config(top_p=1.1).validate()

    def test_validation_positive_values(self):
        with self.assertRaisesRegex(AssertionError, "block_size must be a positive value"):
            iqt.Config(block_size=0).validate()
        with self.assertRaisesRegex(AssertionError, "embed_dim must be a positive value"):
            iqt.Config(embed_dim=0).validate()
        # eval_iters is validated, ensure it's positive
        with self.assertRaisesRegex(AssertionError, "eval_iters must be a positive value"):
            iqt.Config(eval_iters=0).validate()
        with self.assertRaisesRegex(AssertionError, "max_generate_length must be a positive value"):
            iqt.Config(max_generate_length=0).validate()

if __name__ == '__main__':
    unittest.main()

# --- Mocks for QADataset ---
class MockTokenizer:
    def __init__(self, model_file=None):
        self.pad_id_val = 0
        self.bos_id_val = 1
        self.eos_id_val = 2

    def pad_id(self):
        return self.pad_id_val

    def bos_id(self): # Not directly used by QADataset but good for completeness if generate_batch uses it
        return self.bos_id_val
    
    def eos_id(self):
        return self.eos_id_val

    def encode(self, text, out_type=int):
        # Simple mock: returns list of char codes, or predefined sequence for specific texts
        if "Test question 1" in text and "Test answer 1" in text:
            return [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # 10 tokens
        elif "Test question 2" in text and "Test answer 2" in text:
            return [20, 21, 22, 23, 24] # 5 tokens
        elif "Short" in text: # For specific short data test
             return [1,2,3]
        return [ord(c) for c in text]


class TestQADataset(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MockTokenizer()
        self.block_size = 5
        self.step_size = 2
        self.prompt_format = "Q: {question} A: {answer}"
        self.device = "cpu"

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_load_data_success(self, mock_exists, mock_file):
        mock_json_data = '[{"question": "Test question 1", "answer": "Test answer 1"}, {"question": "Test question 2", "answer": "Test answer 2"}]'
        mock_file.return_value.read.return_value = mock_json_data
        
        dataset = iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)
        # Expected: [10-19] from Q1A1, space (mocked as ord(' ')), [20-24] from Q2A2
        # MockTokenizer joins with space, so ord(' ') will be between token lists.
        # Actual logic in QADataset: text = ' '.join(prompts), then one encode call.
        # MockTokenizer.encode for "Q: Test question 1 A: Test answer 1 Q: Test question 2 A: Test answer 2"
        # (assuming space is part of the prompt format or between joined prompts)
        # If "Q: {question} A: {answer}" is the format, then the joined text is complex.
        # Let's simplify: assume load_data processes one item for this test.
        mock_json_data_single = '[{"question": "Test question 1", "answer": "Test answer 1"}]'
        mock_file.return_value.read.return_value = mock_json_data_single
        dataset = iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)
        self.assertEqual(dataset.data, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    @patch('os.path.exists', return_value=False)
    def test_load_data_file_not_found(self, mock_exists):
        with self.assertRaisesRegex(FileNotFoundError, "File not found: dummy_path.json"):
            iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_load_data_json_decode_error(self, mock_exists, mock_file):
        mock_file.return_value.read.return_value = "invalid json"
        with self.assertRaisesRegex(ValueError, "Invalid JSON format in file: dummy_path.json"):
            iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_load_data_key_error(self, mock_exists, mock_file):
        mock_json_data = '[{"question": "q1", "ans": "a1"}]' # "ans" instead of "answer"
        mock_file.return_value.read.return_value = mock_json_data
        with self.assertRaisesRegex(ValueError, "Missing key 'answer' in item 0"):
            iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)
            
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_len_calculation(self, mock_exists, mock_file):
        # Mock data such that self.dataset.data has a specific length
        # Data length 10, step_size 2. Expected len = (10-1)//2 + 1 = 4+1 = 5
        mock_json_data = '[{"question": "Test question 1", "answer": "Test answer 1"}]' # yields 10 tokens
        mock_file.return_value.read.return_value = mock_json_data
        dataset = iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)
        self.assertEqual(len(dataset), 5)

        # Data length 3 (Short), step_size 2. Expected len = (3-1)//2 + 1 = 1+1 = 2
        mock_json_data_short = '[{"question": "Short", "answer": ""}]' # yields 3 tokens
        mock_file.return_value.read.return_value = mock_json_data_short
        dataset_short = iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)
        self.assertEqual(len(dataset_short), 2) # (3-1)//2 + 1 = 2

        # Empty data
        mock_file.return_value.read.return_value = "[]"
        dataset_empty = iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)
        self.assertEqual(len(dataset_empty), 0)


    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_getitem_logic_and_padding(self, mock_exists, mock_file):
        # data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] (10 tokens)
        # block_size = 5, step_size = 2
        mock_json_data = '[{"question": "Test question 1", "answer": "Test answer 1"}]'
        mock_file.return_value.read.return_value = mock_json_data
        dataset = iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)
        
        # idx = 0: start_offset = 0. end_offset = 0 + 5 + 1 = 6. chunk = data[0:6] = [10,11,12,13,14,15]
        # x = chunk[:-1] = [10,11,12,13,14] (len 5)
        # y = chunk[1:]  = [11,12,13,14,15] (len 5)
        x, y = dataset[0]
        self.assertEqual(x.tolist(), [10, 11, 12, 13, 14])
        self.assertEqual(y.tolist(), [11, 12, 13, 14, 15])
        self.assertEqual(x.shape, (self.block_size,))
        self.assertEqual(y.shape, (self.block_size,))

        # idx = 4: start_offset = 4*2 = 8. end_offset = 8 + 5 + 1 = 14.
        # chunk = data[8:14] = data[8:10] = [18,19] (len 2)
        # x = chunk[:-1] = [18] (len 1). Padded: [18, 0, 0, 0, 0] (pad_id=0)
        # y = chunk[1:] = [19] (len 1). Padded: [19, 0, 0, 0, 0]
        x, y = dataset[4] # Last item, len(dataset) is 5. (idx from 0 to 4)
        self.assertEqual(x.tolist(), [18, self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id()])
        self.assertEqual(y.tolist(), [19, self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id()])
        self.assertEqual(x.shape, (self.block_size,))
        self.assertEqual(y.shape, (self.block_size,))

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_getitem_short_data(self, mock_exists, mock_file):
        # data = [1,2,3] (3 tokens from "Short" q/a)
        # block_size = 5, step_size = 2. len = (3-1)//2 + 1 = 2
        mock_json_data_short = '[{"question": "Short", "answer": ""}]'
        mock_file.return_value.read.return_value = mock_json_data_short
        dataset = iqt.QADataset("dummy_path.json", self.mock_tokenizer, self.block_size, self.step_size, self.prompt_format, self.device)
        
        # idx = 0: start_offset = 0. end_offset = 0 + 5 + 1 = 6.
        # chunk = data[0:6] = data[0:3] = [1,2,3] (len 3)
        # x = chunk[:-1] = [1,2] (len 2). Padded: [1,2,0,0,0]
        # y = chunk[1:]  = [2,3] (len 2). Padded: [2,3,0,0,0]
        x, y = dataset[0]
        self.assertEqual(x.tolist(), [1, 2, self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id()])
        self.assertEqual(y.tolist(), [2, 3, self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id(), self.mock_tokenizer.pad_id()])

        # idx = 1: start_offset = 1*2 = 2. end_offset = 2 + 5 + 1 = 8.
        # chunk = data[2:8] = data[2:3] = [3] (len 1)
        # x = chunk[:-1] = [] (len 0). Padded: [0,0,0,0,0]
        # y = chunk[1:] = [] (len 0). Padded: [0,0,0,0,0]
        x, y = dataset[1]
        pad_list = [self.mock_tokenizer.pad_id()] * self.block_size
        self.assertEqual(x.tolist(), pad_list)
        self.assertEqual(y.tolist(), pad_list)


class TestTransformerEncDec(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embed_dim = 16 # Small for testing
        self.num_heads = 2  # Must be divisor of embed_dim
        self.num_layers = 1
        self.dropout = 0.1
        self.max_seq_len = 32 # Max PE length for model
        self.device = "cpu"
        self.pad_token_id = 0 # Assuming pad_id is 0 for these tests

        self.model = iqt.TransformerEncDec(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            device=self.device,
            pad_token_id=self.pad_token_id,
            use_checkpointing=False
        ).to(self.device)
        
        self.model_checkpointing = iqt.TransformerEncDec(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            device=self.device,
            pad_token_id=self.pad_token_id,
            use_checkpointing=True # Enable checkpointing
        ).to(self.device)


    def test_model_instantiation(self):
        self.assertIsInstance(self.model, iqt.TransformerEncDec)
        self.assertEqual(self.model.max_seq_len, self.max_seq_len)

    def test_forward_pass_train(self):
        batch_size = 2
        seq_len = 10 # Shorter than max_seq_len
        
        # Dummy input tensors
        src = torch.randint(1, self.vocab_size, (batch_size, seq_len), device=self.device) # Avoid pad_token_id for simplicity
        tgt = torch.randint(1, self.vocab_size, (batch_size, seq_len), device=self.device)
        
        logits, loss = self.model(src, tgt)
        
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0) # Basic check for a valid loss

    def test_forward_pass_train_checkpointing(self):
        batch_size = 2
        seq_len = 10
        self.model_checkpointing.train() # Set to train mode for checkpointing to be active
        
        src = torch.randint(1, self.vocab_size, (batch_size, seq_len), device=self.device)
        tgt = torch.randint(1, self.vocab_size, (batch_size, seq_len), device=self.device)
        
        logits, loss = self.model_checkpointing(src, tgt)
        
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_forward_pass_inference(self):
        batch_size = 2
        seq_len = 10
        src = torch.randint(1, self.vocab_size, (batch_size, seq_len), device=self.device)
        
        # When tgt is None, model should return memory (encoder output) and None for loss
        memory, loss = self.model(src, tgt=None)
        
        self.assertEqual(memory.shape, (batch_size, seq_len, self.embed_dim))
        self.assertIsNone(loss)

    def test_sequence_truncation_forward(self):
        batch_size = 2
        seq_len_long = self.max_seq_len + 5 # Longer than model's max_seq_len
        
        src = torch.randint(1, self.vocab_size, (batch_size, seq_len_long), device=self.device)
        tgt = torch.randint(1, self.vocab_size, (batch_size, seq_len_long), device=self.device)
        
        logits, loss = self.model(src, tgt)
        
        # Output logits and loss should be based on truncated sequence length (self.max_seq_len)
        self.assertEqual(logits.shape, (batch_size, self.max_seq_len, self.vocab_size))
        self.assertIsNotNone(loss)

    def test_encode_decode_step(self):
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 5 # Number of steps to decode

        src = torch.randint(1, self.vocab_size, (batch_size, src_seq_len), device=self.device)
        
        # Encode
        memory, src_key_padding_mask = self.model.encode(src)
        self.assertEqual(memory.shape, (batch_size, src_seq_len, self.embed_dim))
        self.assertEqual(src_key_padding_mask.shape, (batch_size, src_seq_len))

        # Decode step-by-step (simplified loop)
        # Start with BOS token for each batch item
        tgt_tokens = torch.full((batch_size, 1), self.pad_token_id + 1, dtype=torch.long, device=self.device) # Use a non-pad, non-special token as BOS mock

        for _ in range(tgt_seq_len):
            logits_step = self.model.decode_step(tgt_tokens, memory, src_key_padding_mask)
            self.assertEqual(logits_step.shape, (batch_size, 1, self.vocab_size))
            # In a real generation, we'd sample from logits_step and append to tgt_tokens
            next_token = torch.argmax(logits_step, dim=-1) # Simplistic: take argmax
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
            if tgt_tokens.size(1) > self.model.max_seq_len: # Emulate sliding window in decode_step
                tgt_tokens = tgt_tokens[:, -self.model.max_seq_len:]


        self.assertEqual(tgt_tokens.shape[1], tgt_seq_len + 1) # BOS + generated tokens

    def test_padding_masking(self):
        batch_size = 2
        seq_len = 5
        src = torch.tensor([
            [10, 11, 12, self.pad_token_id, self.pad_token_id],
            [13, 14, self.pad_token_id, self.pad_token_id, self.pad_token_id]
        ], dtype=torch.long, device=self.device)
        
        tgt = torch.tensor([
            [20, 21, 22, self.pad_token_id, self.pad_token_id],
            [23, 24, self.pad_token_id, self.pad_token_id, self.pad_token_id]
        ], dtype=torch.long, device=self.device)

        # This test primarily checks that the forward pass runs with padding
        # and that the loss calculation ignores pad_token_id.
        # A more rigorous test would involve inspecting attention weights, which is complex.
        try:
            _, loss = self.model(src, tgt)
            self.assertIsNotNone(loss)
            # If loss computation did not ignore pad_token_id, loss might be different or error.
            # This is an indirect check.
        except Exception as e:
            self.fail(f"Model forward pass with padding failed: {e}")


class TestGenerateBatch(unittest.TestCase):
    def setUp(self):
        self.mock_config = iqt.Config(
            device="cpu",
            generation_batch_size=1,
            max_generate_length=10,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=0, # Mocked
            bos_token_id=1, # Mocked
            eos_token_id=2  # Mocked
        )

        # Mock Tokenizer
        self.mock_tokenizer = MockTokenizer() # Using the same one as TestQADataset for convenience
        # Ensure config reflects the tokenizer's special IDs
        self.mock_config.pad_token_id = self.mock_tokenizer.pad_id()
        self.mock_config.bos_token_id = self.mock_tokenizer.bos_id()
        self.mock_config.eos_token_id = self.mock_tokenizer.eos_id()


        # Mock Model
        self.mock_model = MagicMock(spec=iqt.TransformerEncDec)
        self.mock_model.device = "cpu"
        self.mock_model.pad_token_id = self.mock_tokenizer.pad_id()
        
        # Mock model.encode output
        # memory_shape: (batch_size, src_len, embed_dim)
        # src_key_padding_mask_shape: (batch_size, src_len)
        # Let embed_dim be small, e.g., 16 (matching TestTransformerEncDec)
        self.embed_dim = 16 
        
        def mock_encode_func(src_tensor):
            batch_size, src_len = src_tensor.shape
            memory = torch.randn(batch_size, src_len, self.embed_dim, device=self.mock_model.device)
            # src_key_padding_mask should be True where src_tensor is pad_id
            src_key_padding_mask = (src_tensor == self.mock_tokenizer.pad_id())
            return memory, src_key_padding_mask
        
        self.mock_model.encode = MagicMock(side_effect=mock_encode_func)

        # Mock model.decode_step output
        # logits_shape: (batch_size, 1, vocab_size) - but generate_batch slices to [batch, vocab_size]
        # Let vocab_size be small, e.g., 10
        self.vocab_size = 10

        # This counter will help simulate EOS generation
        self.decode_step_call_count = 0

        def mock_decode_step_func(tgt_tokens, memory, memory_key_padding_mask):
            batch_size, current_tgt_len = tgt_tokens.shape
            # Simulate logits. Make EOS likely after a few steps.
            logits = torch.rand(batch_size, current_tgt_len, self.vocab_size, device=self.mock_model.device) * 0.1
            
            if self.decode_step_call_count < 3: # Generate some non-EOS tokens
                 # Boost a non-special token (e.g. token id 3)
                logits[:, -1, 3] = 10.0 
            else: # Then generate EOS
                logits[:, -1, self.mock_tokenizer.eos_id()] = 10.0 
            
            self.decode_step_call_count += 1
            return logits

        self.mock_model.decode_step = MagicMock(side_effect=mock_decode_step_func)


    def test_generate_batch_basic(self):
        self.decode_step_call_count = 0 # Reset counter for this test
        prompts = ["Test prompt"]
        
        # Mock tokenizer.decode to make assertion easier
        def mock_decode(token_ids):
            if not token_ids: return ""
            # Simple mock: join numbers with space, handle special tokens
            id_map = {v: k for k,v in {"PAD":0, "BOS":1, "EOS":2, "T3":3}.items()} # T3 for token '3'
            return " ".join([id_map.get(t, str(t)) for t in token_ids])
        self.mock_tokenizer.decode = MagicMock(side_effect=mock_decode)
        
        generated_texts = iqt.generate_batch(self.mock_model, self.mock_tokenizer, prompts, self.mock_config)
        
        self.assertEqual(len(generated_texts), 1)
        
        # Check calls to model methods
        self.mock_model.encode.assert_called_once()
        self.assertTrue(self.mock_model.decode_step.call_count > 0)
        
        # Expected generation: BOS T3 T3 T3 EOS (BOS from prompt, 3xT3 then EOS from decode_step mock)
        # The prompt itself is tokenized by mock_tokenizer.encode("Test prompt") -> [ord(c)...]
        # then BOS is prepended.
        # Let's verify the generated part based on our mock logic:
        # Initial prompt: "Test prompt" -> tokenized by mock_tokenizer.encode
        # Mock encode for "Test prompt" is [ord('T'), ord('e'), ...]
        # BOS is added: [1, ord('T'), ord('e'), ...]
        # Generated sequence: [3, 3, 3, 2] (T3, T3, T3, EOS)
        # So, tokenizer.decode will be called with: [1, ord('T'), ord('e'), ..., 3, 3, 3, 2]
        
        # Get the actual arguments passed to tokenizer.decode
        # The generated_sequences[b] part is [3,3,3,2]
        # encoded_prompts[b] is [1, ord('T'), ...]
        # full_sequence = encoded_prompts_list[batch_idx] + final_token_ids
        
        # Check the decoded output contains expected parts.
        # Example: "BOS 84 101 115 116 32 112 114 111 109 112 116 T3 T3 T3 EOS"
        self.assertIn("BOS", generated_texts[0])
        self.assertIn("T3 T3 T3 EOS", generated_texts[0]) # Check for the generated part
        self.assertTrue(generated_texts[0].endswith("EOS"))


    def test_generate_batch_max_length(self):
        self.decode_step_call_count = 0
        # Modify decode_step to *not* produce EOS, to test max_generate_length
        def mock_decode_no_eos(tgt_tokens, memory, memory_key_padding_mask):
            batch_size, current_tgt_len = tgt_tokens.shape
            logits = torch.rand(batch_size, current_tgt_len, self.vocab_size, device=self.mock_model.device)
            logits[:, -1, 3] = 10.0 # Always generate token '3'
            self.decode_step_call_count +=1
            return logits
        self.mock_model.decode_step.side_effect = mock_decode_no_eos
        
        prompts = ["Another prompt"]
        self.mock_tokenizer.decode = MagicMock(return_value="decoded_text") # Simple decode mock

        generated_texts = iqt.generate_batch(self.mock_model, self.mock_tokenizer, prompts, self.mock_config)
        
        self.assertEqual(self.mock_model.decode_step.call_count, self.mock_config.max_generate_length)
        self.assertEqual(len(generated_texts), 1)
        # The actual content check is tricky without knowing the exact tokenization of "Another prompt"
        # and then appending max_generate_length times token '3'.
        # For now, checking call count and output presence is key.

    def test_generate_batch_empty_prompt_list(self):
        self.decode_step_call_count = 0
        prompts = []
        generated_texts = iqt.generate_batch(self.mock_model, self.mock_tokenizer, prompts, self.mock_config)
        self.assertEqual(len(generated_texts), 0)
        self.mock_model.encode.assert_not_called()
        self.mock_model.decode_step.assert_not_called()
