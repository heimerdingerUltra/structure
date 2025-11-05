from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import os


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    

@dataclass(frozen=True)
class EnvironmentConfig:
    name: Environment
    debug: bool
    mixed_precision: bool
    num_workers: int
    pin_memory: bool
    benchmark: bool
    deterministic: bool
    log_level: str
    checkpoint_dir: Path
    artifact_dir: Path
    
    
_CONFIGS = {
    Environment.DEVELOPMENT: EnvironmentConfig(
        name=Environment.DEVELOPMENT,
        debug=True,
        mixed_precision=False,
        num_workers=0,
        pin_memory=False,
        benchmark=False,
        deterministic=True,
        log_level="DEBUG",
        checkpoint_dir=Path("checkpoints/dev"),
        artifact_dir=Path("artifacts/dev")
    ),
    Environment.PRODUCTION: EnvironmentConfig(
        name=Environment.PRODUCTION,
        debug=False,
        mixed_precision=True,
        num_workers=4,
        pin_memory=True,
        benchmark=True,
        deterministic=False,
        log_level="INFO",
        checkpoint_dir=Path("checkpoints/prod"),
        artifact_dir=Path("artifacts/prod")
    ),
    Environment.STAGING: EnvironmentConfig(
        name=Environment.STAGING,
        debug=False,
        mixed_precision=True,
        num_workers=2,
        pin_memory=True,
        benchmark=True,
        deterministic=False,
        log_level="INFO",
        checkpoint_dir=Path("checkpoints/staging"),
        artifact_dir=Path("artifacts/staging")
    )
}


def get_environment() -> EnvironmentConfig:
    env_name = os.getenv("VOLATILITY_ENV", "development").lower()
    
    try:
        env = Environment(env_name)
    except ValueError:
        env = Environment.DEVELOPMENT
    
    config = _CONFIGS[env]
    
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    
    return config
