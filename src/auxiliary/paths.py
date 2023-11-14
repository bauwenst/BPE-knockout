from pathlib import Path

PATH_SRC = Path(__file__).resolve().parent.parent
PATH_ROOT = PATH_SRC.parent
PATH_DATA = PATH_ROOT / "data"
PATH_DATA_RAW        = PATH_DATA / "raw"
PATH_DATA_COMPRESSED = PATH_DATA / "compressed"
PATH_DATA_MODELBASE  = PATH_DATA / "base-models"
PATH_DATA_OUT        = PATH_DATA / "out"
PATH_DATA_TEMP       = PATH_DATA / "temp"

PATH_DATA           .mkdir(exist_ok=True, parents=False)
PATH_DATA_RAW       .mkdir(exist_ok=True, parents=False)
PATH_DATA_COMPRESSED.mkdir(exist_ok=True, parents=False)
PATH_DATA_MODELBASE .mkdir(exist_ok=True, parents=False)
PATH_DATA_OUT       .mkdir(exist_ok=True, parents=False)
PATH_DATA_TEMP      .mkdir(exist_ok=True, parents=False)
