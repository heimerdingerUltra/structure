import torch.nn as nn
from typing import Dict


class UnifiedModelFactory:
    
    @staticmethod
    def create_tabpfn(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.tabpfn import TabPFN
        return TabPFN(
            n_features=n_features,
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 12),
            n_heads=config.get('n_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_mamba(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.mamba import Mamba
        return Mamba(
            n_features=n_features,
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_layers', 8),
            d_state=config.get('d_state', 16),
            expand=config.get('expand', 2),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_xlstm(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.xlstm import xLSTM
        return xLSTM(
            n_features=n_features,
            hidden_size=config.get('hidden_size', 256),
            n_layers=config.get('n_layers', 4),
            use_mlstm=config.get('use_mlstm', True),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_hypermixer(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.hypermixer import HyperMixer
        return HyperMixer(
            n_features=n_features,
            dim=config.get('dim', 256),
            n_blocks=config.get('n_blocks', 8),
            patch_size=config.get('patch_size', 1),
            expansion_factor=config.get('expansion_factor', 4),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_ttt(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.ttt import TTT
        return TTT(
            n_features=n_features,
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 1024),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_modern_tcn(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.modern_tcn import ModernTCN
        return ModernTCN(
            n_features=n_features,
            channels=config.get('channels', [256, 256, 256, 256]),
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.1),
            use_se=config.get('use_se', True)
        )
    
    @staticmethod
    def create_modern_transformer(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.modern_transformer import ModernTransformer
        return ModernTransformer(
            n_features=n_features,
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 12),
            n_heads=config.get('n_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            attention_type=config.get('attention_type', 'multi_query'),
            use_layer_scale=config.get('use_layer_scale', True),
            pooling=config.get('pooling', 'mean')
        )
    
    @staticmethod
    def create_hybrid_transformer(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.modern_transformer import HybridTransformer
        return HybridTransformer(
            n_features=n_features,
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 12),
            n_heads=config.get('n_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_parallel_transformer(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.modern_transformer import ParallelTransformer
        return ParallelTransformer(
            n_features=n_features,
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 12),
            n_heads=config.get('n_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_modern_resnet(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.modern_resnet import ModernResNet
        return ModernResNet(
            n_features=n_features,
            dim=config.get('dim', 512),
            n_blocks=config.get('n_blocks', 12),
            hidden_dim_ratio=config.get('hidden_dim_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            drop_path_rate=config.get('drop_path_rate', 0.1),
            use_se=config.get('use_se', True),
            use_layer_scale=config.get('use_layer_scale', True)
        )
    
    @staticmethod
    def create_pyramid_resnet(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.modern_resnet import PyramidResNet
        return PyramidResNet(
            n_features=n_features,
            dims=config.get('dims', [256, 384, 512]),
            depths=config.get('depths', [3, 4, 5]),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_densenet(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.modern_resnet import DenseNet
        return DenseNet(
            n_features=n_features,
            init_dim=config.get('init_dim', 256),
            growth_rate=config.get('growth_rate', 64),
            block_config=config.get('block_config', (4, 4, 4)),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_ft_transformer(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.ft_transformer import FTTransformer
        return FTTransformer(
            n_features=n_features,
            d_token=config.get('d_token', 192),
            n_blocks=config.get('n_blocks', 3),
            n_heads=config.get('n_heads', 8),
            d_ffn_factor=config.get('d_ffn_factor', 4/3),
            attention_dropout=config.get('attention_dropout', 0.2),
            ffn_dropout=config.get('ffn_dropout', 0.1),
            residual_dropout=config.get('residual_dropout', 0.0),
            prenormalization=config.get('prenormalization', True),
            use_cls_token=config.get('use_cls_token', True)
        )
    
    @staticmethod
    def create_saint(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.ft_transformer import SAINT
        return SAINT(
            n_features=n_features,
            d_token=config.get('d_token', 192),
            n_blocks=config.get('n_blocks', 6),
            n_heads=config.get('n_heads', 8),
            d_ffn=config.get('d_ffn', 256),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_tabnet(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.ft_transformer import TabNet
        return TabNet(
            n_features=n_features,
            n_d=config.get('n_d', 64),
            n_a=config.get('n_a', 64),
            n_steps=config.get('n_steps', 3),
            gamma=config.get('gamma', 1.3),
            n_independent=config.get('n_independent', 2),
            n_shared=config.get('n_shared', 2)
        )
    
    @staticmethod
    def create(model_name: str, n_features: int, config: Dict) -> nn.Module:
        creators = {
            'tabpfn': UnifiedModelFactory.create_tabpfn,
            'mamba': UnifiedModelFactory.create_mamba,
            'xlstm': UnifiedModelFactory.create_xlstm,
            'hypermixer': UnifiedModelFactory.create_hypermixer,
            'ttt': UnifiedModelFactory.create_ttt,
            'modern_tcn': UnifiedModelFactory.create_modern_tcn,
            'modern_transformer': UnifiedModelFactory.create_modern_transformer,
            'hybrid_transformer': UnifiedModelFactory.create_hybrid_transformer,
            'parallel_transformer': UnifiedModelFactory.create_parallel_transformer,
            'modern_resnet': UnifiedModelFactory.create_modern_resnet,
            'pyramid_resnet': UnifiedModelFactory.create_pyramid_resnet,
            'densenet': UnifiedModelFactory.create_densenet,
            'ft_transformer': UnifiedModelFactory.create_ft_transformer,
            'saint': UnifiedModelFactory.create_saint,
            'tabnet': UnifiedModelFactory.create_tabnet
        }
        
        creator = creators.get(model_name.lower())
        if creator is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        return creator(n_features, config)
