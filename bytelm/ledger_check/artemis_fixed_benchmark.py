import os
import json
import math
import hashlib
import time
import numpy as np
import torch
import torch.nn as nn

GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
    def forward(self, x):
        return self.pos_embeddings(torch.arange(x.size(1), device=x.device).unsqueeze(0))

class ArtemisCausalAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(1)
        attn_weights = torch.softmax(scores, dim=-1)
        return self.out_proj(torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, D))

class ArtemisDecoderBlock(nn.Module):
    def __init__(self, d_model=32, n_heads=2, hidden_dim=64):
        super().__init__()
        self.attn = ArtemisCausalAttention(d_model=d_model, n_heads=n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model))
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ArtemisFixedLanguageModel(nn.Module):
    def __init__(self, vocab_size=256, d_model=32, n_layers=2, max_seq_len=128):
        super().__init__()
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([ArtemisDecoderBlock(d_model=d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.byte_embedding(x) + self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

class HardenedDatasetLoader:
    def __init__(self, workspace_dir="data_splits"):
        self.workspace = workspace_dir
        self._safely_write_defaults()
        self.train_docs, self.train_hash = self._read_and_hash("train.txt")
        self.val_docs, self.val_hash = self._read_and_hash("val.txt")
        self.test_docs, self.test_hash = self._read_and_hash("test.txt")
        self._verify_leakage_boundaries()
        self._compute_unigram_baseline()
    def _safely_write_defaults(self):
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
        defaults = {
            "train.txt": "sample document line one for optimization loop validation.\nline two provides further byte variations.",
            "val.txt": "held out validation sets isolate performance metrics safely.",
            "test.txt": "completely independent test lines check for true generalization capabilities."
        }
        for name, text in defaults.items():
            path = os.path.join(self.workspace, name)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
    def _read_and_hash(self, filename):
        path = os.path.join(self.workspace, filename)
        with open(path, "rb") as f:
            raw_content = f.read()
        return [line.decode('utf-8') for line in raw_content.split(b'\n') if line.strip()], hashlib.sha256(raw_content).hexdigest()
    def _verify_leakage_boundaries(self):
        tr, va, te = set(self.train_docs), set(self.val_docs), set(self.test_docs)
        if tr.intersection(va) or tr.intersection(te) or va.intersection(te):
            raise ValueError("FATAL ERROR: Cross-partition sequence leakage discovered.")
    def _compute_unigram_baseline(self):
        counts = np.zeros(256)
        total_bytes = 0
        for doc in self.train_docs:
            b_arr = np.frombuffer(doc.encode('utf-8'), dtype=np.uint8)
            counts += np.bincount(b_arr, minlength=256)
            total_bytes += len(b_arr)
        self.smoothed_probs = (counts + 1.0) / (total_bytes + 256.0)

def slice_document_sequences(docs, max_window=128):
    slices = []
    for doc in docs:
        raw_bytes = [b for b in doc.encode('utf-8')]
        if len(raw_bytes) <= 1: continue
        for i in range(0, len(raw_bytes) - 1, max_window - 1):
            chunk = raw_bytes[i : i + max_window]
            if len(chunk) > 1: slices.append(chunk)
    return slices

def score_partition_metrics(model, window_slices, smoothed_probs):
    model.eval()
    total_neural_nll, total_unigram_nll, scored_bytes = 0.0, 0.0, 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for chunk in window_slices:
            for i in range(1, len(chunk)):
                inp = torch.tensor([chunk[:i]], dtype=torch.long, device=DEVICE)
                targ = torch.tensor([chunk[i]], dtype=torch.long, device=DEVICE)
                total_neural_nll += loss_fn(model(inp)[:, -1, :], targ).item()
                total_unigram_nll -= np.log2(smoothed_probs[chunk[i]])
                scored_bytes += 1
    return total_neural_nll / (scored_bytes * math.log(2)), total_unigram_nll / scored_bytes, scored_bytes

def run_reproducible_experiment():
    dataset = HardenedDatasetLoader()
    tr_slices = slice_document_sequences(dataset.train_docs)
    va_slices = slice_document_sequences(dataset.val_docs)
    te_slices = slice_document_sequences(dataset.test_docs)
    model = ArtemisFixedLanguageModel(n_layers=2).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    total_train_bytes_processed, optimizer_steps, best_val_bpb, best_checkpoint, best_epoch_idx, val_history_log = 0, 0, float('inf'), None, 0, []
    start_time = time.time()
    for epoch in range(40):
        model.train()
        for chunk in tr_slices:
            inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=DEVICE)
            targ = torch.tensor([chunk[1:]], dtype=torch.long, device=DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(inp).view(-1, 256), targ.view(-1))
            loss.backward()
            optimizer.step()
            total_train_bytes_processed += inp.size(1)
            optimizer_steps += 1
        val_bpb, _, _ = score_partition_metrics(model, va_slices, dataset.smoothed_probs)
        val_history_log.append(val_bpb)
        if val_bpb < best_val_bpb:
            best_val_bpb, best_epoch_idx = val_bpb, epoch
            best_checkpoint = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_checkpoint)
    test_model_bpb, test_unigram_bpb, test_bytes_count = score_partition_metrics(model, te_slices, dataset.smoothed_probs)
    verifiable_ledger = {
        "timestamp_unix_sec": time.time(),
        "environment_hardware": str(DEVICE),
        "configuration": {"experiment_seed": GLOBAL_SEED, "training_epochs_limit": 40, "architecture_param_count": param_count},
        "repository_verified_hashes": {"train_sha256": dataset.train_hash, "val_sha256": dataset.val_hash, "test_sha256": dataset.test_hash},
        "optimization_telemetry": {"total_optimizer_steps_executed": optimizer_steps, "total_training_bytes_processed": total_train_bytes_processed, "elapsed_wall_clock_seconds": time.time() - start_time, "best_epoch_selected": best_epoch_idx, "validation_bpb_history": val_history_log},
        "evaluation_metrics": {"identical_scored_bytes_denominator": test_bytes_count, "smoothed_unigram_baseline_bpb": float(test_unigram_bpb), "artemis_model_final_test_bpb": float(test_model_bpb)}
    }
    print("\n--- REPRODUCIBLE LEDGER VERIFICATION COMPLETE ---")
    print(json.dumps(verifiable_ledger, indent=2))

if __name__ == "__main__":
    run_reproducible_experiment()
