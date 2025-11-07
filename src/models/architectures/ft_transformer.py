import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import RMSNorm
from .activations import get_activation


class NumericalFeatureTokenizer(nn.Module):
    
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.randn(n_features, d_token))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        return x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FeatureTokenizer(nn.Module):
    
    def __init__(self, n_features: int, d_token: int, bias: bool = True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        if bias:
            self.bias = nn.Parameter(torch.randn(n_features, d_token))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) * self.weight.unsqueeze(0)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        return x


class CLSToken(nn.Module):
    
    def __init__(self, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_token))
        
    def forward(self, batch_size: int) -> torch.Tensor:
        return self.weight.unsqueeze(0).expand(batch_size, 1, -1)


class FTTransformerBlock(nn.Module):
    
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        d_ffn: int,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        prenormalization: bool = True
    ):
        super().__init__()
        
        self.prenormalization = prenormalization
        
        self.attention_norm = RMSNorm(d_token)
        self.attention = nn.MultiheadAttention(
            d_token,
            n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attention_residual_dropout = nn.Dropout(residual_dropout)
        
        self.ffn_norm = RMSNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(ffn_dropout)
        )
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenormalization:
            x_attn = self.attention_norm(x)
            x = x + self.attention_residual_dropout(
                self.attention(x_attn, x_attn, x_attn, need_weights=False)[0]
            )
            x = x + self.ffn_residual_dropout(self.ffn(self.ffn_norm(x)))
        else:
            x = self.attention_norm(
                x + self.attention_residual_dropout(
                    self.attention(x, x, x, need_weights=False)[0]
                )
            )
            x = self.ffn_norm(x + self.ffn_residual_dropout(self.ffn(x)))
        
        return x


class FTTransformer(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        d_ffn_factor: float = 4/3,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        prenormalization: bool = True,
        kv_compression_ratio: float = None,
        use_cls_token: bool = True
    ):
        super().__init__()
        
        assert d_token % n_heads == 0
        
        self.tokenizer = FeatureTokenizer(n_features, d_token, bias=True)
        
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = CLSToken(d_token)
        
        d_ffn = int(d_token * d_ffn_factor)
        
        self.blocks = nn.ModuleList([
            FTTransformerBlock(
                d_token,
                n_heads,
                d_ffn,
                attention_dropout,
                ffn_dropout,
                residual_dropout,
                prenormalization
            )
            for _ in range(n_blocks)
        ])
        
        self.prenormalization = prenormalization
        if prenormalization:
            self.head_norm = RMSNorm(d_token)
        
        self.head = nn.Linear(d_token, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        
        if self.use_cls_token:
            cls_token = self.cls_token(x.size(0))
            x = torch.cat([cls_token, x], dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        if self.prenormalization:
            x = self.head_norm(x)
        
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        
        return self.head(x).squeeze(-1)


class SAINT(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        d_token: int = 192,
        n_blocks: int = 6,
        n_heads: int = 8,
        d_ffn: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        
        self.inter_sample_blocks = nn.ModuleList([
            FTTransformerBlock(d_token, n_heads, d_ffn, dropout, dropout)
            for _ in range(n_blocks)
        ])
        
        self.intra_sample_blocks = nn.ModuleList([
            FTTransformerBlock(d_token, n_heads, d_ffn, dropout, dropout)
            for _ in range(n_blocks)
        ])
        
        self.norm = RMSNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        
        for inter_block, intra_block in zip(self.inter_sample_blocks, self.intra_sample_blocks):
            x_transposed = x.transpose(0, 1)
            x_transposed = inter_block(x_transposed)
            x = x_transposed.transpose(0, 1)
            
            x = intra_block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        
        return self.head(x).squeeze(-1)


class TabNet(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        momentum: float = 0.98,
        mask_type: str = "sparsemax"
    ):
        super().__init__()
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.mask_type = mask_type
        
        self.bn = nn.BatchNorm1d(n_features, momentum=momentum)
        
        self.initial_splitter = nn.Sequential(
            nn.Linear(n_features, n_d + n_a),
            nn.BatchNorm1d(n_d + n_a, momentum=momentum)
        )
        
        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()
        
        for step in range(n_steps):
            transformer = nn.Sequential(
                nn.Linear(n_a if step > 0 else n_d + n_a, n_d + n_a),
                nn.BatchNorm1d(n_d + n_a, momentum=momentum),
                nn.GLU(dim=-1)
            )
            self.feat_transformers.append(transformer)
            
            attention = nn.Sequential(
                nn.Linear(n_a if step > 0 else n_d + n_a, n_features),
                nn.BatchNorm1d(n_features, momentum=momentum)
            )
            self.att_transformers.append(attention)
        
        self.head = nn.Linear(n_d, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        
        prior_scales = torch.ones(x.shape[0], x.shape[1], device=x.device)
        
        M_loss = 0
        att = self.initial_splitter(x)
        
        steps_output = []
        
        for step in range(self.n_steps):
            M = self.att_transformers[step](att)
            M = torch.mul(M, prior_scales)
            
            M = F.softmax(M, dim=-1)
            
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )
            
            prior_scales = torch.mul(self.gamma - M, prior_scales)
            
            masked_x = torch.mul(M, x)
            
            out = self.feat_transformers[step](masked_x)
            d = F.relu(out[:, :self.n_d])
            steps_output.append(d)
            
            att = out[:, self.n_d:]
        
        out = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        
        return self.head(out).squeeze(-1)
