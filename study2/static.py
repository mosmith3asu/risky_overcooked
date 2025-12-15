import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root_dir = os.path.join(_current_dir.split("risky_overcooked")[0] , "risky_overcooked")


HUMANDATA_DIR = os.path.join(_current_dir, "human_data")

PROCESSED_DIR = os.path.join(_current_dir, "human_data", "processed")
PROCESSED_COND0_DIR = os.path.join(PROCESSED_DIR, "cond_0")
PROCESSED_COND1_DIR = os.path.join(PROCESSED_DIR, "cond_1")
PROCESSED_REJECT_DIR = os.path.join(PROCESSED_DIR, "rejected")

RAW_DIR = os.path.join(_current_dir, "human_data", "data")
RAW_COND0_DIR = os.path.join(RAW_DIR, "cond_0")
RAW_COND1_DIR = os.path.join(RAW_DIR, "cond_1")
AGENT_DIR =_proj_root_dir+ r'/src/risky_overcooked_webserver/server/static/assets/agents/RiskSensitiveAI/'