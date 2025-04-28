from pathlib import Path

# Donot give here local system related absolute path because it'll throw error when app will run on the production server.
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")











