"""
Mess3 Non-Ergodic Transformer Experiment
=========================================

Setup:
- 3 Mess3 processes with different parameters = 3 ergodic components
- Each training sequence comes entirely from ONE component  -> non-ergodic dataset
- Train a small transformer via next-token prediction
- Analyze residual stream geometry via PCA / CEV
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SEED         = 42
SEQ_LEN      = 16        # tokens per sequence (excluding BOS)
N_SEQUENCES  = 50_000
N_LAYERS     = 2
D_MODEL      = 64
N_HEADS      = 2
D_MLP        = 256
CONTEXT_LEN  = SEQ_LEN + 1   # +1 for BOS token
N_STEPS      = 8_000
BATCH_SIZE   = 256
LR           = 5e-4
LOG_EVERY    = 250           # print loss every N steps

# Vocabulary: 0, 1, 2 are Mess3 tokens; 3 is BOS
VOCAB_SIZE   = 4
BOS_TOKEN    = 3

# The 3 Mess3 processes (ergodic components).
# Each sequence in our dataset comes entirely from ONE of these.
#
# Parameter choice: found by grid search maximising min pairwise KL divergence.

PROCESS_PARAMS = [
    (0.95, 0.02),  # P1: very strong structure, belief concentrated at vertices
    (0.90, 0.05),  # P2: strong structure, intermediate spread
    (0.85, 0.12),  # P3: moderate structure, broadest belief geometry
]

# Save directory for plots
os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────
# PART 1 — MESS3 DATA GENERATION
# ─────────────────────────────────────────────

def make_mess3_matrices(alpha, x):
    """
    Build the 3 labeled transition matrices for a Mess3 process.

    Standard Mess3 construction:
        beta = (1 - alpha) / 2
        y    = 1 - 2*x

    T[k][i][j] = probability of emitting token k AND
                 transitioning from hidden state i to hidden state j.

    The net transition T = T[0]+T[1]+T[2] is row-stochastic.
    Stationary distribution is uniform: pi = [1/3, 1/3, 1/3].
    """
    beta = (1 - alpha) / 2
    y    = 1 - 2 * x

    T0 = np.array([
        [alpha*y, beta*x, beta*x],
        [alpha*x, beta*y, beta*x],
        [alpha*x, beta*x, beta*y],
    ])

    T1 = np.array([
        [beta*y, alpha*x, beta*x],
        [beta*x, alpha*y, beta*x],
        [beta*x, alpha*x, beta*y],
    ])

    T2 = np.array([
        [beta*y, beta*x, alpha*x],
        [beta*x, beta*y, alpha*x],
        [beta*x, beta*x, alpha*y],
    ])

    T = np.stack([T0, T1, T2])  # shape: (3 tokens, 3 states, 3 states)

    # Quick sanity check: net matrix should be row-stochastic
    net = T.sum(axis=0)
    assert np.allclose(net.sum(axis=1), 1.0, atol=1e-9), "Net transition not row-stochastic!"

    return T


def generate_sequence(T, length, rng):
    """
    Sample one token sequence from a Mess3 process.

    At each step from hidden state s:
      - Joint probability over (token k, next state s') is T[k, s, s']
      - Sample (k, s') from this joint distribution
      - Emit token k, move to state s'

    Start from stationary distribution (uniform for Mess3).
    """
    state  = rng.integers(3)   # uniform stationary start
    tokens = []

    for _ in range(length):
        # Flatten joint (token, next_state) distribution from current state
        # T[:, state, :] has shape (3 tokens, 3 next_states) -> flatten to 9
        probs = T[:, state, :].flatten()
        probs = probs / probs.sum()   # normalize (should already sum to 1)

        idx        = rng.choice(9, p=probs)
        token      = idx // 3       # which token was emitted
        next_state = idx  % 3       # which state to move to

        tokens.append(token)
        state = next_state

    return tokens


def build_dataset(process_params, n_sequences, seq_len, seed):
    """
    Build the non-ergodic dataset.

    Non-ergodicity: the dataset mixes K=3 ergodic Mess3 processes.
    Each sequence is generated entirely by ONE process.
    The model never sees which process generated its input —
    it must infer this from the token statistics.

    Returns:
        sequences : (n_sequences, seq_len+1)  int array, BOS prepended
        labels    : (n_sequences,)             which process (0/1/2)
    """
    rng  = np.random.default_rng(seed)
    K    = len(process_params)
    n_each = n_sequences // K

    sequences_list = []
    labels_list    = []

    for proc_id, (alpha, x) in enumerate(process_params):
        T = make_mess3_matrices(alpha, x)
        print(f"  Generating {n_each} sequences for P{proc_id+1} "
              f"(alpha={alpha}, x={x})")

        for _ in range(n_each):
            tokens = generate_sequence(T, seq_len, rng)
            seq    = [BOS_TOKEN] + tokens   # prepend BOS
            sequences_list.append(seq)
            labels_list.append(proc_id)

    sequences = np.array(sequences_list, dtype=np.int64)
    labels    = np.array(labels_list,    dtype=np.int64)

    # Shuffle so processes are interleaved during training
    idx = rng.permutation(len(sequences))
    return sequences[idx], labels[idx]


def compute_belief_states(T, sequence):
    """
    Compute ground-truth belief states for one sequence.

    belief[t] = distribution over hidden states after observing
                tokens sequence[0..t-1]  (i.e., t tokens seen).

    Standard Bayesian filtering update for a hidden Markov model:
        eta(x1:t) = eta(empty) @ T(x1) @ ... @ T(xt)   [then normalize]

    Returns: (seq_len+1, 3) array — belief at each position 0..seq_len
    """
    # Initial belief = stationary distribution (uniform for Mess3)
    belief = np.ones(3) / 3.0

    beliefs = [belief.copy()]   # belief before seeing any token

    for token in sequence:
        if token == BOS_TOKEN:
            beliefs.append(belief.copy())
            continue
        # Update: eta -> eta @ T[token], then normalize
        unnorm  = belief @ T[token]   # shape (3,)
        belief  = unnorm / unnorm.sum()
        beliefs.append(belief.copy())

    return np.array(beliefs)   # (len(sequence)+1, 3)


# ─────────────────────────────────────────────
# PART 2 — SMALL TRANSFORMER
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model,     d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        # Compute Q, K, V all at once
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)               # each: (B, L, n_heads, d_head)
        q = q.transpose(1, 2)                      # (B, n_heads, L, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = self.d_head ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, L, L)
        mask  = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn  = attn.masked_fill(mask, float('-inf'))
        attn  = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """One transformer block: LayerNorm -> Attention -> residual,
                               LayerNorm -> MLP       -> residual."""

    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp,   d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention with residual
        x = x + self.mlp(self.ln2(x))    # MLP with residual
        return x


class SmallGPT(nn.Module):
    """
    Small decoder-only transformer.

    Residual stream: x flows through blocks unchanged except for
    the residual additions — this is where belief state geometry lives.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_mlp, n_layers, context_len):
        super().__init__()
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(context_len, d_model)
        self.blocks   = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights simply
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, return_residuals=False):
        """
        tokens: (B, L) long tensor

        If return_residuals=True, also return a list of residual stream
        activations at each layer (for PCA analysis later).
        """
        B, L = tokens.shape
        pos  = torch.arange(L, device=tokens.device)

        # Token + positional embeddings -> initial residual stream
        x = self.tok_emb(tokens) + self.pos_emb(pos)

        residuals = []
        if return_residuals:
            residuals.append(x.detach().cpu())   # after embedding, before block 1

        for block in self.blocks:
            x = block(x)
            if return_residuals:
                residuals.append(x.detach().cpu())

        x      = self.ln_f(x)
        logits = self.head(x)

        if return_residuals:
            return logits, residuals   # residuals[i] = (B, L, D) after layer i
        return logits


# ─────────────────────────────────────────────
# PART 3 — TRAINING
# ─────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_model(model, dataset, n_steps, batch_size, lr, device):
    """
    Train the transformer on next-token prediction.

    For each sequence [BOS, t1, t2, ..., t16]:
      input  = [BOS, t1, ..., t15]   (first 16 tokens)
      target = [t1,  t2, ..., t16]   (next token at each position)
    """
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    losses = []
    step   = 0

    print(f"\nTraining for {n_steps} steps...")
    print(f"  Batch size: {batch_size}, LR: {lr}")
    print(f"  Model: {n_layers} layers, d_model={D_MODEL}, {N_HEADS} heads\n")

    while step < n_steps:
        for sequences, _ in loader:
            if step >= n_steps:
                break

            sequences = sequences.to(device)       # (B, seq_len+1)
            inp    = sequences[:, :-1]              # (B, seq_len)   — input tokens
            target = sequences[:, 1:]              # (B, seq_len)   — next tokens

            logits = model(inp)                    # (B, seq_len, vocab_size)
            loss   = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                target.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % LOG_EVERY == 0 or step == 1:
                avg = np.mean(losses[-LOG_EVERY:])
                print(f"  Step {step:>5}/{n_steps}  |  Loss: {avg:.4f}")

    print(f"\nFinal loss: {np.mean(losses[-200:]):.4f}")
    return losses


# ─────────────────────────────────────────────
# PART 4 — QUICK ANALYSIS (PCA / CEV)
# ─────────────────────────────────────────────

def get_residual_activations(model, dataset, device, n_samples=2000):
    """
    Run the model on a subset of sequences and collect
    residual stream activations at every layer and position.

    Returns:
        activations : list of (n_samples, seq_len, d_model) arrays
                      one per layer (layer 0 = after embedding, layer 1/2 = after each block)
        labels      : (n_samples,) process labels
    """
    loader  = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()

    all_residuals = None   # will be list of lists
    all_labels    = []

    collected = 0
    with torch.no_grad():
        for sequences, labels in loader:
            if collected >= n_samples:
                break

            inp = sequences[:, :-1].to(device)   # (B, seq_len)
            _, residuals = model(inp, return_residuals=True)
            # residuals is a list of n_layers+1 tensors, each (B, seq_len, d_model)

            if all_residuals is None:
                all_residuals = [[] for _ in residuals]

            for layer_idx, r in enumerate(residuals):
                all_residuals[layer_idx].append(r.cpu().numpy())

            all_labels.append(labels.numpy())
            collected += len(sequences)

    # Stack along batch dimension
    activations = [np.concatenate(r, axis=0)[:n_samples] for r in all_residuals]
    labels_out  = np.concatenate(all_labels)[:n_samples]

    return activations, labels_out


def compute_cev(activations_2d):
    """
    Compute Cumulative Explained Variance (CEV) via PCA.

    activations_2d : (N, d_model) — N data points, each of dimension d_model
    Returns:
        cev    : (d_model,) — CEV at each number of components
        eff_dim: int        — min components for 95% variance
    """
    # Center the data
    X      = activations_2d - activations_2d.mean(axis=0, keepdims=True)
    # SVD (more numerically stable than eig for this purpose)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    variance = s ** 2
    cev      = np.cumsum(variance) / variance.sum()
    eff_dim  = int(np.searchsorted(cev, 0.95)) + 1
    return cev, eff_dim


# ─────────────────────────────────────────────
# PART 5 — PLOTTING
# ─────────────────────────────────────────────

def plot_training_loss(losses):
    """Plot training loss over steps."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Smooth with a simple moving average
    window  = 50
    smooth  = np.convolve(losses, np.ones(window)/window, mode='valid')
    steps   = np.arange(len(smooth)) + window

    ax.plot(losses, color='lightblue', alpha=0.4, label='raw loss')
    ax.plot(steps, smooth, color='steelblue', linewidth=2, label=f'smoothed (w={window})')

    # Reference: entropy of uniform over {0,1,2} = log(3) ≈ 1.099
    ax.axhline(np.log(3), color='gray', linestyle='--', label='H[uniform] = log 3')

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training Loss — Mess3 Non-Ergodic Transformer")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/figure_0_training_loss.png", dpi=150)
    plt.show()
    print("Saved: results/figure_0_training_loss.png")


def plot_cev_by_layer(activations, labels):
    """
    Plot CEV curves for each layer's residual stream.
    Each curve = full dataset.
    """
    n_layers = len(activations)
    colors   = plt.cm.viridis(np.linspace(0, 1, n_layers))
    layer_names = ['Embedding'] + [f'After Layer {i+1}' for i in range(n_layers - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    eff_dims = []

    for layer_idx, (acts, color, name) in enumerate(zip(activations, colors, layer_names)):
        # acts: (N, seq_len, d_model) — flatten all positions
        acts_2d    = acts.reshape(-1, acts.shape[-1])
        cev, eff_d = compute_cev(acts_2d)
        eff_dims.append(eff_d)

        axes[0].plot(np.arange(1, len(cev)+1), cev, color=color, label=name)

    # Reference lines
    axes[0].axhline(0.95, color='red', linestyle='--', linewidth=1, label='95% threshold')
    axes[0].set_xlabel("Number of PCA components")
    axes[0].set_ylabel("Cumulative Explained Variance")
    axes[0].set_title("CEV by Layer (all positions pooled)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim([1, 30])

    # Effective dimensionality bar chart
    axes[1].bar(layer_names, eff_dims, color=colors)
    axes[1].set_ylabel("Effective dimensionality (95% CEV)")
    axes[1].set_title("Effective Dimensionality per Layer")
    axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig("results/figure_1_cev_by_layer.png", dpi=150)
    plt.show()
    print("Saved: results/figure_1_cev_by_layer.png")


def plot_cev_by_position(activations, labels):
    """
    For the last layer's residual stream, plot effective dimensionality
    as a function of context position.

    Key prediction: early positions are high-D (process unknown),
    late positions collapse toward ~2D (process identified).
    """
    last_layer = activations[-1]    # (N, seq_len, d_model)
    seq_len    = last_layer.shape[1]
    positions  = np.arange(seq_len)

    eff_dims_all = []      # pooled across all processes
    eff_dims_per = [[] for _ in range(len(PROCESS_PARAMS))]   # per process

    for t in positions:
        acts_t     = last_layer[:, t, :]       # (N, d_model)
        _, eff_d   = compute_cev(acts_t)
        eff_dims_all.append(eff_d)

        for proc_id in range(len(PROCESS_PARAMS)):
            mask       = labels == proc_id
            acts_proc  = last_layer[mask, t, :]
            _, eff_d_p = compute_cev(acts_proc)
            eff_dims_per[proc_id].append(eff_d_p)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(positions, eff_dims_all, 'k-o', linewidth=2, label='All processes')
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for proc_id, (eff_d_p, c) in enumerate(zip(eff_dims_per, colors)):
        ax.plot(positions, eff_d_p, '--s', color=c, linewidth=1.5,
                label=f'P{proc_id+1} only')

    # Reference: 2D = single Mess3 belief simplex
    ax.axhline(2, color='gray', linestyle=':', label='2D (single Mess3 simplex)')

    ax.set_xlabel("Context position")
    ax.set_ylabel("Effective dimensionality (95% CEV)")
    ax.set_title("How Residual Stream Dimensionality Changes With Context\n"
                 "(Last layer — non-ergodic Mess3 dataset)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("results/figure_2_eff_dim_by_position.png", dpi=150)
    plt.show()
    print("Saved: results/figure_2_eff_dim_by_position.png")


def plot_pca_2d(activations, labels, position=-1):
    """
    Plot the first 2 PCA components of the last-layer residual stream
    at a specific context position, colored by process identity.
    """
    last_layer = activations[-1]             # (N, seq_len, d_model)
    acts_pos   = last_layer[:, position, :]  # (N, d_model)

    # PCA
    X      = acts_pos - acts_pos.mean(axis=0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    proj   = X @ Vt[:2].T                   # (N, 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ['tab:blue', 'tab:orange', 'tab:green']
    names   = [f'P{i+1} (α={a}, x={x})' for i, (a,x) in enumerate(PROCESS_PARAMS)]

    for proc_id, (c, name) in enumerate(zip(colors, names)):
        mask = labels == proc_id
        ax.scatter(proj[mask, 0], proj[mask, 1], c=c, s=6, alpha=0.4, label=name)

    pos_label = "last" if position == -1 else str(position)
    ax.set_title(f"Residual Stream — Last Layer, Position {pos_label}\n"
                 f"(First 2 PCA components, colored by process)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"results/pca_2d_pos{pos_label}.png", dpi=150)
    plt.show()
    print(f"Saved: results/pca_2d_pos{pos_label}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RESIDUAL STREAM GEOMETRY ANALYSIS
#
# Method:
#   1. Compute ground truth belief states η(x1:t) for each sequence/position
#   2. Check if belief states are linearly decodable from residual stream
#   3. Visualize residual stream geometry vs ground truth belief simplex
#   4. Compare early vs late context positions
#
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ground_truth_beliefs(sequences, labels, process_params):
    """
    Compute the ground truth belief state η(x1:t) at every context position.

    Exact Bayesian filtering: after observing tokens x1,...,xt the belief is:
        η(x1:t) = η(∅) T(x1)...T(xt) / normalisation

    For Mess3, η(∅) = [1/3, 1/3, 1/3]  (uniform stationary distribution).

    sequences : (N, CONTEXT_LEN)  — includes BOS at index 0
    labels    : (N,)              — which process each sequence came from

    Returns:
        beliefs : (N, input_len, 3)
                  beliefs[i, t] = belief state at input position t for sequence i
                  t=0 → BOS, no update yet → uniform
                  t=k → after observing k Mess3 tokens
    """
    # Build transition matrices for each process
    T_mats = [make_mess3_matrices(a, x) for a, x in process_params]

    N          = len(sequences)
    input_len  = sequences.shape[1] - 1    # sequences[:, :-1] is the input
    beliefs    = np.zeros((N, input_len, 3), dtype=np.float32)

    for i in range(N):
        T      = T_mats[labels[i]]
        belief = np.ones(3) / 3.0           # uniform start

        for t in range(input_len):
            beliefs[i, t] = belief          # belief BEFORE seeing token at position t
            token = sequences[i, t]         # token at input position t
            if token == BOS_TOKEN:
                continue                    # BOS doesn't update belief
            unnorm = belief @ T[token]
            belief = unnorm / unnorm.sum()

    return beliefs                          # (N, input_len, 3)


def linear_decode_beliefs(acts_layer, beliefs, positions=None):
    """
    Fit a linear map from residual stream activations to belief states.
    Uses closed-form least squares.  No neural networks — purely linear.

    acts_layer : (N, input_len, d_model)
    beliefs    : (N, input_len, 3)
    positions  : list of positions to evaluate, or None for all

    Returns:
        rmse_per_pos : (n_positions,)   — RMSE at each context position
        r2_per_pos   : (n_positions,)   — R² at each context position
    """
    if positions is None:
        positions = list(range(acts_layer.shape[1]))

    rmse_list, r2_list = [], []

    for t in positions:
        X = acts_layer[:, t, :]     # (N, d_model)  — activations
        Y = beliefs[:, t, :]        # (N, 3)        — ground truth beliefs

        # Train/test split (80/20) to avoid overfitting
        n_train = int(0.8 * len(X))
        X_tr, Y_tr = X[:n_train], Y[:n_train]
        X_te, Y_te = X[n_train:], Y[n_train:]

        # Closed-form linear regression: W = (X^T X)^{-1} X^T Y
        W, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
        Y_pred     = X_te @ W

        # RMSE
        rmse = np.sqrt(np.mean((Y_te - Y_pred) ** 2))
        # R²
        ss_res = np.sum((Y_te - Y_pred) ** 2)
        ss_tot = np.sum((Y_te - Y_te.mean(axis=0)) ** 2)
        r2     = 1 - ss_res / (ss_tot + 1e-12)

        rmse_list.append(rmse)
        r2_list.append(r2)

    return np.array(rmse_list), np.array(r2_list)


def plot_belief_decoding(rmse_per_pos, r2_per_pos, save_dir):
    """
    Plot RMSE and R² of linear belief-state decoding vs context position.
    Good R² = residual stream linearly encodes the belief geometry.
    """
    positions = np.arange(len(rmse_per_pos))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(positions, rmse_per_pos, 'o-', color='steelblue')
    axes[0].set_xlabel("Context position")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Belief-state decoding RMSE\n(lower = better linear encoding)")

    axes[1].plot(positions, r2_per_pos, 'o-', color='darkorange')
    axes[1].axhline(1.0, color='gray', linestyle='--', label='perfect')
    axes[1].set_xlabel("Context position")
    axes[1].set_ylabel("R²")
    axes[1].set_title("Belief-state decoding R²\n(higher = better linear encoding)")
    axes[1].legend()

    plt.suptitle("Can we linearly decode the belief state from residual stream?",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "figure_6_belief_decoding.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def plot_ground_truth_simplices(process_params, save_dir, n_steps=3000):
    """
    Visualize the ground truth belief geometry for each Mess3 process.
    Run each process forward and plot the trajectory of belief states
    in the 2-simplex (projected to 2D via the standard simplex embedding).

    The simplex embedding: map (b0, b1, b2) -> 2D using
        x = b1 + 0.5*b2,  y = (sqrt(3)/2)*b2
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(process_params), figsize=(5 * len(process_params), 4))

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for idx, ((alpha, x), ax, color) in enumerate(zip(process_params, axes, colors)):
        T      = make_mess3_matrices(alpha, x)
        belief = np.ones(3) / 3.0
        state  = rng.integers(3)
        pts    = []

        for _ in range(n_steps):
            pts.append(belief.copy())
            # sample next token + state
            probs  = T[:, state, :].flatten()
            probs /= probs.sum()
            i      = rng.choice(9, p=probs)
            token  = i // 3
            state  = i  % 3
            unnorm = belief @ T[token]
            belief = unnorm / unnorm.sum()

        pts = np.array(pts)
        # Simplex 2D embedding
        px = pts[:, 1] + 0.5 * pts[:, 2]
        py = (np.sqrt(3) / 2) * pts[:, 2]

        ax.scatter(px, py, c=color, s=3, alpha=0.3)
        ax.set_title(f"P{idx+1}: α={alpha}, x={x}\nGround-truth belief simplex")
        ax.set_xlabel("Simplex x"); ax.set_ylabel("Simplex y")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 0.95)

    plt.suptitle("Ground-truth belief geometries for each Mess3 process", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "figure_5_ground_truth_simplices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def plot_residual_colored_by_belief(acts_layer, beliefs, labels,
                                    position, save_dir, process_params=PROCESS_PARAMS):
    """
    PCA scatter of residual stream at a given position, with two colorings:
      Left : colored by process identity (P1/P2/P3)
      Right: colored by belief state coordinate b0 (probability of hidden state 0)

    This reveals whether the geometry in the residual stream mirrors
    the belief simplex structure.
    """
    X = acts_layer[:, position, :]      # (N, d_model)
    b = beliefs[:, position, :]         # (N, 3)

    # PCA projection to 2D
    Xc       = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj     = Xc @ Vt[:2].T           # (N, 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    proc_colors = ['tab:blue', 'tab:orange', 'tab:green']
    names       = [f'P{i+1}' for i in range(len(process_params))]

    # Left: colored by process
    for pid, (c, name) in enumerate(zip(proc_colors, names)):
        mask = labels == pid
        axes[0].scatter(proj[mask, 0], proj[mask, 1],
                        c=c, s=8, alpha=0.4, label=name)
    axes[0].set_title(f"Position {position}: colored by process")
    axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
    axes[0].legend(markerscale=3, fontsize=9)

    # Right: colored by belief coordinate b0
    sc = axes[1].scatter(proj[:, 0], proj[:, 1],
                         c=b[:, 0], cmap='viridis', s=8, alpha=0.5)
    plt.colorbar(sc, ax=axes[1], label='Belief b₀ (prob. hidden state 0)')
    axes[1].set_title(f"Position {position}: colored by belief b₀")
    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")

    pos_label = "early" if position <= 2 else "late"
    plt.suptitle(f"Residual stream geometry at {pos_label} context (pos={position})",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, f"residual_vs_belief_pos{position}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def plot_process_decoding_accuracy(acts_layer, labels, save_dir):
    """
    Train a linear classifier to predict which process (P1/P2/P3)
    generated each sequence, from residual stream at each position.

    If accuracy is high at late positions → the transformer has learned
    to distinguish ergodic components. If low at early positions →
    process identity isn't yet resolved.
    """
    from numpy.linalg import lstsq

    input_len = acts_layer.shape[1]
    n_classes = len(np.unique(labels))
    accs      = []

    # One-hot encode labels
    Y_onehot = np.eye(n_classes)[labels]     # (N, 3)

    for t in range(input_len):
        X      = acts_layer[:, t, :]          # (N, d_model)
        n_tr   = int(0.8 * len(X))
        X_tr, Y_tr = X[:n_tr], Y_onehot[:n_tr]
        X_te,       = X[n_tr:],
        Y_te_labels = labels[n_tr:]

        W, _, _, _ = lstsq(X_tr, Y_tr, rcond=None)
        logits      = X_te @ W                # (N_te, 3)
        preds       = logits.argmax(axis=1)
        acc         = (preds == Y_te_labels).mean()
        accs.append(acc)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(input_len), accs, 'o-', color='purple')
    ax.axhline(1/n_classes, color='gray', linestyle='--', label='chance (33%)')
    ax.axhline(1.0,          color='gray', linestyle=':',  label='perfect')
    ax.set_xlabel("Context position")
    ax.set_ylabel("Linear decoding accuracy")
    ax.set_title("Can we linearly decode which process generated the sequence?\n"
                 "(measures when the transformer resolves ergodic component identity)")
    ax.legend(); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    path = os.path.join(save_dir, "figure_9_process_decoding.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")
    return np.array(accs)


def plot_per_process_belief_geometry(acts_layer, beliefs, labels,
                                     position, save_dir, process_params=PROCESS_PARAMS):
    """
    For each process separately, project residual stream activations
    to 2D and overlay the decoded belief state as color.

    This directly compares the residual stream triangle (for each process)
    to its ground truth belief simplex.
    """
    fig, axes = plt.subplots(1, len(process_params), figsize=(5 * len(process_params), 4))
    proc_colors = ['tab:blue', 'tab:orange', 'tab:green']

    for pid, (ax, color) in enumerate(zip(axes, proc_colors)):
        mask  = labels == pid
        X     = acts_layer[mask, position, :]   # (N_pid, d_model)
        b     = beliefs[mask, position, :]       # (N_pid, 3)

        Xc       = X - X.mean(axis=0)
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj     = Xc @ Vt[:2].T

        sc = ax.scatter(proj[:, 0], proj[:, 1],
                        c=b[:, 0], cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(sc, ax=ax, label='belief b₀')
        alpha_p, x_p = process_params[pid]
        ax.set_title(f"P{pid+1} (α={alpha_p}, x={x_p})\nResidual stream @ pos {position}")
        ax.set_xlabel("PC 1 (within process)")
        ax.set_ylabel("PC 2 (within process)")

    plt.suptitle("Per-process residual stream geometry (colored by belief b₀)",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, f"per_process_geometry_pos{position}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# GEOMETRY ANALYSIS MAIN BLOCK
# ─────────────────────────────────────────────

def run_geometry_analysis(model, val_dataset, sequences, labels, device):
    """
    Run all geometry analyses. Call this after training is complete.
    Saves all figures to results/geometry/
    """
    GEOMETRY_DIR = "results/geometry"
    os.makedirs(GEOMETRY_DIR, exist_ok=True)

    print("\n" + "═" * 60)
    print("RESIDUAL STREAM GEOMETRY ANALYSIS")
    print("═" * 60)

    # ── Step A: Get residual stream activations ──────────────────
    print("\n[A] Collecting residual stream activations...")
    val_seqs   = sequences[int(0.9 * len(sequences)):]
    val_labels = labels[int(0.9 * len(labels)):]

    activations, act_labels = get_residual_activations(
        model, val_dataset, device, n_samples=3000
    )
    # activations[i]: (N, input_len, d_model) — layer i
    last_layer = activations[-1]    # (N, input_len, d_model)
    print(f"    Activations shape: {last_layer.shape}")

    # ── Step B: Compute ground truth belief states ───────────────
    print("\n[B] Computing ground truth belief states...")
    val_seqs_sub   = val_seqs[:3000]
    val_labels_sub = val_labels[:3000]
    beliefs = compute_ground_truth_beliefs(val_seqs_sub, val_labels_sub, PROCESS_PARAMS)
    print(f"    Belief states shape: {beliefs.shape}")
    print(f"    Sample belief at pos 8: {beliefs[0, 8].round(3)}")

    # ── Step C: Ground truth simplices ───────────────────────────
    print("\n[C] Plotting ground truth belief simplices...")
    plot_ground_truth_simplices(PROCESS_PARAMS, GEOMETRY_DIR)

    # ── Step D: Linear decoding of belief states ─────────────────
    print("\n[D] Testing linear decodability of belief states from residual stream...")
    positions    = list(range(last_layer.shape[1]))
    rmse, r2     = linear_decode_beliefs(last_layer, beliefs, positions)

    print(f"    R² at position  1: {r2[1]:.3f}  (early — process unknown)")
    print(f"    R² at position 15: {r2[-1]:.3f} (late  — process resolved)")
    print(f"    Mean R² across all positions: {r2.mean():.3f}")

    plot_belief_decoding(rmse, r2, GEOMETRY_DIR)

    # ── Step E: Residual geometry colored by process & belief ────
    print("\n[E] Plotting residual stream geometry (early vs late)...")
    plot_residual_colored_by_belief(last_layer, beliefs, act_labels,
                                    position=1,  save_dir=GEOMETRY_DIR)   # early
    plot_residual_colored_by_belief(last_layer, beliefs, act_labels,
                                    position=15, save_dir=GEOMETRY_DIR)   # late

    # ── Step F: Process identity decoding ───────────────────────
    print("\n[F] Testing linear decoding of process identity across positions...")
    accs = plot_process_decoding_accuracy(last_layer, act_labels, GEOMETRY_DIR)
    print(f"    Accuracy at position  1: {accs[1]:.2%}")
    print(f"    Accuracy at position 15: {accs[-1]:.2%}")

    # ── Step G: Per-process geometry ─────────────────────────────
    print("\n[G] Per-process residual stream geometry at late context...")
    plot_per_process_belief_geometry(last_layer, beliefs, act_labels,
                                     position=15, save_dir=GEOMETRY_DIR)

    print(f"\nGeometry analysis done. All figures saved to ./{GEOMETRY_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_layers = N_LAYERS
    print(f"Device: {device}\n")

    # ── Generate data ────────────────────────────────────────
    print("=" * 55)
    print("STEP 1: Generating non-ergodic Mess3 dataset")
    print("=" * 55)

    sequences, labels = build_dataset(
        process_params=PROCESS_PARAMS,
        n_sequences=N_SEQUENCES,
        seq_len=SEQ_LEN,
        seed=SEED,
    )
    print(f"\n  Dataset shape : {sequences.shape}")
    print(f"  Label counts  : {np.bincount(labels)}")

    n_train       = int(0.9 * len(sequences))
    train_dataset = SequenceDataset(sequences[:n_train], labels[:n_train])
    val_dataset   = SequenceDataset(sequences[n_train:], labels[n_train:])
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Build model ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2: Building transformer")
    print("=" * 55)

    model = SmallGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_mlp=D_MLP,
        n_layers=N_LAYERS,
        context_len=CONTEXT_LEN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Train ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 3: Training")
    print("=" * 55)

    losses = train_model(model, train_dataset, N_STEPS, BATCH_SIZE, LR, device)
    plot_training_loss(losses)

    # ── Basic geometry plots ─────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 4: Basic geometry (CEV, PCA)")
    print("=" * 55)

    activations, act_labels = get_residual_activations(
        model, val_dataset, device, n_samples=3000
    )

    layer_names = ['Embedding'] + [f'Layer {i+1}' for i in range(N_LAYERS)]
    print("\n  Effective dimensionality per layer:")
    for name, acts in zip(layer_names, activations):
        _, eff_d = compute_cev(acts.reshape(-1, acts.shape[-1]))
        print(f"    {name}: {eff_d}D")

    plot_cev_by_layer(activations, act_labels)
    plot_cev_by_position(activations, act_labels)
    plot_pca_2d(activations, act_labels, position=1)
    plot_pca_2d(activations, act_labels, position=-1)

    # ── Geometry analysis ────────────────────────────────────
    run_geometry_analysis(model, val_dataset, sequences, labels, device)
