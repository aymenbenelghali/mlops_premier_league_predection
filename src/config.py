import os
from dataclasses import dataclass


@dataclass
class Settings:
    data_dir: str = os.getenv("DATA_DIR", os.path.join("data"))
    models_dir: str = os.getenv("MODELS_DIR", os.path.join("models"))
    rolling_window: int = int(os.getenv("ROLLING_WINDOW", "5"))
    seasons_back: int = int(os.getenv("SEASONS_BACK", "4"))
    league_code: str = os.getenv("LEAGUE_CODE", "E0")  # Premier League
    # Model filenames
    model_path: str = os.getenv("MODEL_PATH", os.path.join("models", "xgb_multioutput.joblib"))
    metadata_path: str = os.getenv("METADATA_PATH", os.path.join("models", "feature_metadata.joblib"))


settings = Settings()


