import os
from pathlib import Path

PATH_PACKAGE = Path(__file__).resolve().parent.parent
PATH_SRC     = PATH_PACKAGE.parent

IS_EDITABLE_INSTALL = PATH_SRC.name == "src"
PATH_CWD = Path(os.getcwd())

# Paths to the data that are delivered with the package.
PATH_PACKAGE_DATA = PATH_PACKAGE / "_data"
PATH_MODELBASE  = PATH_PACKAGE_DATA / "base-models"
PATH_MORPHOLOGY = PATH_PACKAGE_DATA / "morphology"

# Paths to the data that are NOT delivered with the package by default.
if IS_EDITABLE_INSTALL:  # Comes pre-delivered with a place to store data.
    PATH_ROOT          = PATH_SRC.parent
    PATH_EXTERNAL_DATA = PATH_ROOT / "data"
    IS_RUNNING_INSIDE_PROJECT = PATH_CWD.is_relative_to(PATH_ROOT)
else:  # Store data (compressed corpora etc.) in the package itself, since it is the central place where all projects on the machine could access it. The alternative is the user cache, but that's on the same disk anyway.
    PATH_ROOT          = None
    PATH_EXTERNAL_DATA = PATH_PACKAGE_DATA
    IS_RUNNING_INSIDE_PROJECT = False
assert not(IS_RUNNING_INSIDE_PROJECT and not IS_EDITABLE_INSTALL)

PATH_DATA_COMPRESSED = PATH_EXTERNAL_DATA / "compressed"
PATH_DATA_TEMP       = PATH_EXTERNAL_DATA / "temp"
PATH_EXTERNAL_DATA  .mkdir(exist_ok=True, parents=False)
PATH_DATA_COMPRESSED.mkdir(exist_ok=True, parents=False)
PATH_DATA_TEMP      .mkdir(exist_ok=True, parents=False)

# Output is either written to BPE-knockout/data/out (not bpe_knockout/_data/out) if it's an editable install and you are in the project, else just a default file tree under CWD.
if IS_RUNNING_INSIDE_PROJECT:
    PATH_DATA_OUT = PATH_EXTERNAL_DATA / "out"
else:
    PATH_DATA_OUT = PATH_CWD / "data" / "out" / "bpe_knockout"

PATH_DATA_OUT.mkdir(exist_ok=True, parents=True)
