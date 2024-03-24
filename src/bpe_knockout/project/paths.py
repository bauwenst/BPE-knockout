from pathlib import Path

PATH_PACKAGE = Path(__file__).resolve().parent.parent
PATH_SRC  = PATH_PACKAGE.parent
PATH_ROOT = PATH_SRC.parent

# Data delivered with the package.
PATH_PACKAGE_DATA = PATH_PACKAGE / "_data"
PATH_MODELBASE  = PATH_PACKAGE_DATA / "base-models"
PATH_MORPHOLOGY = PATH_PACKAGE_DATA / "morphology"

# External data, outside of the package (only applicable to editable installs).
PATH_DATA = PATH_ROOT / "data"
PATH_DATA_COMPRESSED = PATH_DATA / "compressed"
PATH_DATA_OUT        = PATH_DATA / "out"
PATH_DATA_TEMP       = PATH_DATA / "temp"

PATH_DATA           .mkdir(exist_ok=True, parents=False)
PATH_DATA_COMPRESSED.mkdir(exist_ok=True, parents=False)
PATH_DATA_OUT       .mkdir(exist_ok=True, parents=False)
PATH_DATA_TEMP      .mkdir(exist_ok=True, parents=False)
