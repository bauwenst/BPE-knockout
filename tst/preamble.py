# First we make an output path that doesn't belong to the main package. Users of the package in src/ have no use for output generated in tst/.
from bpe_knockout.project.paths import *

# Output is either written to BPE-knockout/data/out (not bpe_knockout/_data/out) if it's an editable install and you are in the project, else just a default file tree under CWD.
if IS_RUNNING_INSIDE_PROJECT:
    PATH_EXPERIMENTS_OUT = PATH_EXTERNAL_DATA / "out"
else:
    PATH_EXPERIMENTS_OUT = PATH_CWD / "data" / "out"
PATH_EXPERIMENTS_OUT.mkdir(exist_ok=True, parents=True)

# Point Fiject to this folder
from fiject import setFijectOutputFolder
setFijectOutputFolder(PATH_EXPERIMENTS_OUT)

# And TkTkT if you need it
from tktkt import setTkTkToutputRoot
from tktkt.paths import PathManager
setTkTkToutputRoot(PATH_EXPERIMENTS_OUT)
OutputPaths = PathManager("bpe-knockout")
