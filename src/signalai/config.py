from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path('/mnt/DATA/signalai_data')
DATASETS_DIR = DATA_DIR / 'datasets'

TEMP_DIR = Path('/dev/shm/.temp_signalai')
TEMP_DIR.mkdir(exist_ok=True)

CONFIGS_DIR = BASE_DIR / "configs"
BASE_DATA_DIR = BASE_DIR / 'data'
TASKS_DIR = DATA_DIR / 'task_data'

DEVICE = "cuda"
