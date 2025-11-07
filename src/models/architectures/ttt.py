import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, mini_batch_size: int = 16, 
                 n_inner_steps: int = 1, lr_inner: float = 0.01):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.mini_batch_size = mini_batch_size
        self.n_inner_steps = n_inner_steps
        self.lr_inner = lr_inner
        
        self.W_ln = nn.Linear(d_model, d_model, bias=False)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.theta = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        
    def inner_loop_update(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        reconstruction = X @ theta
        loss = F.mse_loss(reconstruction, X)
        
        grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
        theta_new = theta - self.lr_inner * grad
        
        return theta_new
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        theta_updated = self.theta
        
        if self.training:
            for _ in range(self.n_inner_steps):
                theta_updated = self.inner_loop_update(x, theta_updated)
        
        x_ln = self.W_ln(x)
        context = x_ln @ theta_updated
        
        q = self.W_q(context)
        k = self.W_k(x)
        v = self.W_v(x)
        
        d_k = self.d_model // self.n_heads
        q = q.view(batch_size, seq_len, self.n_heads, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        out = self.W_o(out)
        
        return out


class TTTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1, mini_batch_size: int = 16):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.ttt = TTTLayer(d_model, n_heads, mini_batch_size)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.ttt(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class TTT(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        mini_batch_size: int = 16
    ):
        super().__init__()
        
        self.embed = nn.Linear(n_features, d_model)
        
        self.blocks = nn.ModuleList([
            TTTBlock(d_model, n_heads, d_ff, dropout, mini_batch_size)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        
        x = x.unsqueeze(1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        x = x.mean(dim=1)
        
        return self.head(x).squeeze(-1)
