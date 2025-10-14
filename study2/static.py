import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_current_dir, "human_data", "data")
RAW_COND0_DIR = os.path.join(RAW_DIR, "cond_0")
RAW_COND1_DIR = os.path.join(RAW_DIR, "cond_1")
PROCESSED_DIR = os.path.join(_current_dir, "processed")