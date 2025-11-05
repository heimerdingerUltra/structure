import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Callable, Dict
import os


class DistributedOrchestrator:
    
    def __init__(
        self,
        backend: str = 'nccl',
        init_method: str = 'env://',
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        self.backend = backend
        self.world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        self.rank = rank or int(os.environ.get('RANK', 0))
        
        if self.world_size > 1:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=self.world_size,
                rank=self.rank
            )
        
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
    def wrap_model(self, model: nn.Module, device_ids: Optional[list] = None) -> nn.Module:
        if self.world_size == 1:
            return model
        
        device_ids = device_ids or [self.local_rank]
        
        model = model.to(device_ids[0])
        
        model = DDP(
            model,
            device_ids=device_ids,
            output_device=device_ids[0],
            find_unused_parameters=False
        )
        
        return model
    
    def create_dataloader(self, dataset, batch_size: int, shuffle: bool = True, **kwargs):
        sampler = None
        
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            shuffle = False
        
        from torch.utils.data import DataLoader
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs
        )
    
    def reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if self.world_size == 1:
            return metrics
        
        reduced = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value).cuda(self.local_rank)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = (tensor / self.world_size).item()
        
        return reduced
    
    def is_main_process(self) -> bool:
        return self.rank == 0
    
    def barrier(self):
        if self.world_size > 1:
            dist.barrier()
    
    def cleanup(self):
        if self.world_size > 1:
            dist.destroy_process_group()


class PipelineParallel(nn.Module):
    
    def __init__(self, model: nn.Module, split_size: int, devices: list):
        super().__init__()
        
        self.split_size = split_size
        self.devices = devices
        
        layers = list(model.children())
        n_layers = len(layers)
        n_devices = len(devices)
        
        layers_per_device = n_layers // n_devices
        
        self.stages = nn.ModuleList()
        
        for i, device in enumerate(devices):
            start_idx = i * layers_per_device
            end_idx = (i + 1) * layers_per_device if i < n_devices - 1 else n_layers
            
            stage = nn.Sequential(*layers[start_idx:end_idx]).to(device)
            self.stages.append(stage)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = torch.split(x, self.split_size, dim=0)
        
        outputs = []
        
        for split in splits:
            current = split
            
            for stage, device in zip(self.stages, self.devices):
                current = current.to(device)
                current = stage(current)
            
            outputs.append(current)
        
        return torch.cat(outputs, dim=0)


class GradientAccumulator:
    
    def __init__(self, model: nn.Module, accumulation_steps: int):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def backward(self, loss: torch.Tensor):
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.current_step += 1
        
        return self.current_step % self.accumulation_steps == 0
    
    def should_step(self) -> bool:
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self):
        self.current_step = 0


class ZeroRedundancyOptimizer:
    
    def __init__(self, optimizer_class, params, world_size: int, rank: int, **kwargs):
        self.world_size = world_size
        self.rank = rank
        
        param_list = list(params)
        
        params_per_rank = len(param_list) // world_size
        start_idx = rank * params_per_rank
        end_idx = start_idx + params_per_rank if rank < world_size - 1 else len(param_list)
        
        self.local_params = param_list[start_idx:end_idx]
        
        self.optimizer = optimizer_class(self.local_params, **kwargs)
        
        self.param_groups = self.optimizer.param_groups
    
    def step(self):
        self.optimizer.step()
        
        if self.world_size > 1:
            for param in self.local_params:
                dist.broadcast(param.data, src=self.rank)
    
    def zero_grad(self):
        self.optimizer.zero_grad()


class TensorParallel:
    
    @staticmethod
    def split_column(weight: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        splits = torch.chunk(weight, world_size, dim=0)
        return splits[rank].contiguous()
    
    @staticmethod
    def split_row(weight: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        splits = torch.chunk(weight, world_size, dim=1)
        return splits[rank].contiguous()
    
    @staticmethod
    def gather_column(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered, tensor)
        
        return torch.cat(gathered, dim=0)
    
    @staticmethod
    def gather_row(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered, tensor)
        
        return torch.cat(gathered, dim=1)


class ColumnParallelLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        
        assert out_features % world_size == 0
        
        self.in_features = in_features
        self.out_features = out_features // world_size
        self.world_size = world_size
        self.rank = rank
        
        self.weight = nn.Parameter(torch.randn(self.out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, self.bias)
        
        return output


class RowParallelLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        
        assert in_features % world_size == 0
        
        self.in_features = in_features // world_size
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        self.weight = nn.Parameter(torch.randn(out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight)
        
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        output = output + self.bias
        
        return output


class MixedPrecisionManager:
    
    def __init__(self, enabled: bool = True, init_scale: float = 2.**16):
        self.enabled = enabled
        
        if enabled:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
        else:
            self.scaler = None
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_(self, optimizer):
        if self.enabled:
            self.scaler.unscale_(optimizer)
    
    def autocast(self):
        if self.enabled:
            return torch.cuda.amp.autocast()
        else:
            return torch.cuda.amp.autocast(enabled=False)


class ActivationCheckpointing:
    
    @staticmethod
    def checkpoint_sequential(functions, segments, input, **kwargs):
        if isinstance(functions, nn.Sequential):
            functions = list(functions.children())
        
        segment_size = len(functions) // segments
        
        def run_function(start, end, functions):
            def forward(input):
                for j in range(start, end):
                    input = functions[j](input)
                return input
            return forward
        
        if segments == 1:
            return run_function(0, len(functions), functions)(input)
        
        from torch.utils.checkpoint import checkpoint
        
        for i in range(segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < segments - 1 else len(functions)
            
            input = checkpoint(run_function(start, end, functions), input, **kwargs)
        
        return input
