import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass
import time


@dataclass
class PredictionResult:
    prediction: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    model_version: str = ""
    
    def to_dict(self) -> Dict:
        result = {
            'prediction': self.prediction.tolist(),
            'latency_ms': self.latency_ms,
            'model_version': self.model_version
        }
        if self.uncertainty is not None:
            result['uncertainty'] = self.uncertainty.tolist()
        return result


class InferencePipeline:
    
    def __init__(
        self,
        model: torch.nn.Module,
        scaler,
        feature_extractor,
        device: str = 'cuda',
        batch_size: int = 512,
        enable_onnx: bool = False
    ):
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.feature_extractor = feature_extractor
        self.device = device
        self.batch_size = batch_size
        self.enable_onnx = enable_onnx
        
        if enable_onnx:
            self._export_onnx()
        
    def _export_onnx(self):
        dummy_input = torch.randn(1, self.model.n_features).to(self.device)
        
        onnx_path = Path("models/model.onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        X = self.feature_extractor.transform(df)
        return X
    
    def predict(
        self,
        df: pd.DataFrame,
        return_uncertainty: bool = False
    ) -> PredictionResult:
        
        start_time = time.time()
        
        X = self.preprocess(df)
        
        predictions = []
        
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i + self.batch_size]
            batch_tensor = torch.FloatTensor(batch).to(self.device)
            
            with torch.no_grad():
                pred = self.model(batch_tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        
        latency = (time.time() - start_time) * 1000
        
        return PredictionResult(
            prediction=predictions,
            latency_ms=latency,
            model_version="1.0.0"
        )
    
    def predict_single(self, sample: Dict) -> float:
        df = pd.DataFrame([sample])
        result = self.predict(df)
        return result.prediction[0]
    
    def batch_predict(
        self,
        df_list: list,
        parallel: bool = True
    ) -> list:
        
        if not parallel:
            return [self.predict(df) for df in df_list]
        
        results = []
        for df in df_list:
            result = self.predict(df)
            results.append(result)
        
        return results


class ModelServer:
    
    def __init__(
        self,
        model_registry_path: str,
        device: str = 'cuda'
    ):
        self.registry_path = Path(model_registry_path)
        self.device = device
        self.loaded_models = {}
        
    def load_model(self, model_name: str, version: str = "latest"):
        from src.models.registry import ModelRegistry
        
        registry = ModelRegistry(str(self.registry_path))
        
        model, metadata, scaler, feature_names = registry.load_model(
            model_name,
            version
        )
        
        self.loaded_models[model_name] = {
            'model': model.to(self.device),
            'metadata': metadata,
            'scaler': scaler,
            'feature_names': feature_names
        }
    
    def predict(
        self,
        model_name: str,
        data: Union[pd.DataFrame, Dict]
    ) -> PredictionResult:
        
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_info = self.loaded_models[model_name]
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        from src.data.advanced_features import AdvancedFeatures
        fe = AdvancedFeatures()
        fe.scaler = model_info['scaler']
        fe.feature_names = model_info['feature_names']
        
        pipeline = InferencePipeline(
            model=model_info['model'],
            scaler=model_info['scaler'],
            feature_extractor=fe,
            device=self.device
        )
        
        return pipeline.predict(data)
    
    def health_check(self) -> Dict:
        return {
            'status': 'healthy',
            'loaded_models': list(self.loaded_models.keys()),
            'device': self.device
        }


class BatchInference:
    
    @staticmethod
    def process_file(
        input_path: str,
        output_path: str,
        pipeline: InferencePipeline,
        chunk_size: int = 10000
    ):
        
        df = pd.read_excel(input_path) if input_path.endswith('.xlsx') else pd.read_csv(input_path)
        
        all_predictions = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            result = pipeline.predict(chunk)
            all_predictions.append(result.prediction)
        
        df['predicted_iv'] = np.concatenate(all_predictions)
        
        if output_path.endswith('.xlsx'):
            df.to_excel(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
    
    @staticmethod
    def process_stream(
        pipeline: InferencePipeline,
        data_stream,
        callback
    ):
        
        for data in data_stream:
            result = pipeline.predict(data)
            callback(result)
