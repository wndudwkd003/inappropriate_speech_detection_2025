import os

def init_hub_env(token: str):
    os.environ["HF_TOKEN"] = token
