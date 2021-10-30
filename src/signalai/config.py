from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
PIPELINE_SAVE_PATH = PROJECT_DIR / "data"

CONFIG_DIR = PROJECT_DIR / "configs"

DEVICE = "cuda"
