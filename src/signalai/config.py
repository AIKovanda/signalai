from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

CONFIGS_DIR = BASE_DIR / 'configs'
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

DATASETS_DIR = DATA_DIR / 'datasets'
TASKS_DIR = DATA_DIR / 'task_data'
EVALUATION_DIR = DATA_DIR / 'eval'

TASKS_DIR.mkdir(exist_ok=True)
EVALUATION_DIR.mkdir(exist_ok=True)
