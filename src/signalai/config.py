from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

CONFIGS_DIR = BASE_DIR / 'configs'

DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

TASKS_DIR = DATA_DIR / 'task_data'
TASKS_DIR.mkdir(exist_ok=True)
