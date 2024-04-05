# Output is either written to BPE-knockout/data/out if it's an editable install and you are in the project, else just a default file tree under CWD.
from bpe_knockout.project.paths import IS_INSIDE_PROJECT, PATH_CWD, PATH_EXTERNAL_DATA
if IS_INSIDE_PROJECT:
    PATH_DATA_OUT = PATH_EXTERNAL_DATA / "out"
else:
    PATH_DATA_OUT = PATH_CWD / "data" / "out" / "bpe_knockout"
PATH_DATA_OUT.mkdir(exist_ok=True, parents=True)

# Point Fiject to this folder
from fiject import setFijectOutputFolder
setFijectOutputFolder(PATH_DATA_OUT)
