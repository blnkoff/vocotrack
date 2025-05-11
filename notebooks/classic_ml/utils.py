from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data"

RAW_DATA_PATH = DATA_PATH / "raw"
CSV_PATH = RAW_DATA_PATH / "ASMDD.csv"
AUDIO_PATH = RAW_DATA_PATH / "ASMDD"