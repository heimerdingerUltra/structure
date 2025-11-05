import torch
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import joblib
from dataclasses import dataclass, asdict


@dataclass
class ModelMetadata:
    model_name: str
    model_type: str
    version: str
    timestamp: str
    n_features: int
    hyperparameters: Dict
    metrics: Dict
    git_hash: Optional[str] = None
    tags: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        return cls(**data)


class ModelRegistry:
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_dir / "index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self) -> None:
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(
        self,
        model: torch.nn.Module,
        metadata: ModelMetadata,
        scaler: Any = None,
        feature_names: list = None
    ) -> str:
        
        version = metadata.version
        model_dir = self.registry_dir / metadata.model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), model_dir / "model.pt")
        
        if scaler is not None:
            joblib.dump(scaler, model_dir / "scaler.pkl")
        
        if feature_names is not None:
            with open(model_dir / "features.json", 'w') as f:
                json.dump(feature_names, f)
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        model_key = f"{metadata.model_name}:{version}"
        self.index[model_key] = {
            'path': str(model_dir),
            'timestamp': metadata.timestamp,
            'metrics': metadata.metrics
        }
        self._save_index()
        
        return model_key
    
    def load_model(
        self,
        model_name: str,
        version: str = "latest",
        model_class = None
    ) -> tuple:
        
        if version == "latest":
            version = self._get_latest_version(model_name)
        
        model_dir = self.registry_dir / model_name / version
        
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = ModelMetadata.from_dict(json.load(f))
        
        if model_class is None:
            raise ValueError("model_class must be provided")
        
        model = model_class(
            n_features=metadata.n_features,
            **metadata.hyperparameters
        )
        model.load_state_dict(torch.load(model_dir / "model.pt"))
        
        scaler = None
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        feature_names = None
        features_path = model_dir / "features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
        
        return model, metadata, scaler, feature_names
    
    def _get_latest_version(self, model_name: str) -> str:
        model_path = self.registry_dir / model_name
        if not model_path.exists():
            raise ValueError(f"Model {model_name} not found")
        
        versions = [d.name for d in model_path.iterdir() if d.is_dir()]
        if not versions:
            raise ValueError(f"No versions found for {model_name}")
        
        versions.sort(reverse=True)
        return versions[0]
    
    def list_models(self) -> Dict:
        return self.index
    
    def get_model_info(self, model_name: str, version: str = "latest") -> ModelMetadata:
        if version == "latest":
            version = self._get_latest_version(model_name)
        
        model_dir = self.registry_dir / model_name / version
        with open(model_dir / "metadata.json", 'r') as f:
            return ModelMetadata.from_dict(json.load(f))
    
    def delete_model(self, model_name: str, version: str) -> None:
        model_dir = self.registry_dir / model_name / version
        
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        model_key = f"{model_name}:{version}"
        if model_key in self.index:
            del self.index[model_key]
            self._save_index()


def create_version_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
