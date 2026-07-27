import torch
import torch.nn as nn
import torch.nn.functional as F

class RETAIN(nn.Module):
    def __init__(self, cat_cardinalities, num_herb, emb_size=128, dropout=0.5, threshold=0.8, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_herb = num_herb
        self.emb_size = emb_size
        self.threshold = threshold
        self.input_len = int(sum(cat_cardinalities))
        self.pad_id = self.input_len
        accum = torch.tensor(cat_cardinalities, dtype=torch.long)
        if accum.numel() > 1:
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(accum[:-1], dim=0)])
        else:
            offsets = torch.zeros(1, dtype=torch.long)
        self.register_buffer("offsets", offsets)
        self.embedding = nn.Embedding(self.input_len + 1, emb_size, padding_idx=self.pad_id)
        self.dropout = nn.Dropout(dropout)
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)
        self.output = nn.Linear(emb_size, num_herb)
        self.reg_head = nn.Sequential(nn.Linear(emb_size, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_herb))

    def build_tokens(self, X_cat):
        X_cat = X_cat.long().to(self.device)
        num_cols = X_cat.shape[1]
        ofs = self.offsets[:num_cols].unsqueeze(0).to(self.device)
        return X_cat + ofs

    def forward_tokens(self, tokens):
        tokens = tokens.to(self.device).long()
        mask = tokens != self.pad_id
        emb = self.embedding(tokens)
        emb = self.dropout(emb)
        g, _ = self.alpha_gru(emb)
        h, _ = self.beta_gru(emb)
        attn_g = self.alpha_li(g).squeeze(-1)
        attn_g = attn_g.masked_fill(~mask, -1e9)
        attn_g = F.softmax(attn_g, dim=1)
        attn_h = torch.tanh(self.beta_li(h))
        c = (attn_g.unsqueeze(-1) * attn_h * emb).sum(dim=1)
        logits = self.output(F.relu(c))
        prob = torch.sigmoid(logits)
        binary_mask = (prob > self.threshold).float().detach()
        values = F.relu(self.reg_head(c) * binary_mask)
        return {"pred_logits": logits, "pred_values": values}

class TCM_RETAIN(nn.Module):
    def __init__(self, cat_cardinalities, num_herb, emb_size=128, dropout=0.5, threshold=0.8, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retain = RETAIN(cat_cardinalities, num_herb, emb_size, dropout, threshold, self.device)

    def forward(self, X, bank=None, gat=None):
        X1 = X[1]
        num_cols = min(X1.shape[1], self.retain.offsets.numel())
        tokens = self.retain.build_tokens(X1[:, :num_cols])
        return self.retain.forward_tokens(tokens)
