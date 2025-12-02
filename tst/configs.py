# # from modest.languages.dutch import Dutch_Celex
# from modest.languages.german import German_Celex
# from modest.languages.english import English_Celex
from modest.languages.german import German_MorphyNet_Inflections, German_MorphyNet_Derivations
from modest.languages.english import English_MorphyNet_Inflections, English_MorphyNet_Derivations
from modest.transformations.combine import ChainedModestDatasets

from bpe_knockout.project.config import *


def setupDutch() -> ProjectConfig:
    return ProjectConfig(
        language_name="Dutch",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-nl.txt"),
        morphologies=Dutch_Celex(legacy=True),  # TODO: THEY KILLED CELEX OPEN ACCESS NOOOOOOOO
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-nl-clean"),
        reweighter=LINEAR_WEIGHTER
    )


def setupGerman() -> ProjectConfig:
    return ProjectConfig(
        language_name="German",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-de.txt"),
        # morphologies=German_Celex(legacy=True),
        morphologies=ChainedModestDatasets([German_MorphyNet_Inflections(), German_MorphyNet_Derivations()]),
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-de-clean"),
        reweighter=LINEAR_WEIGHTER
    )


def setupEnglish() -> ProjectConfig:
    return ProjectConfig(
        language_name="English",
        lemma_counts=OscarWordFile(PATH_DATA_COMPRESSED / "words_oscar-en.txt"),
        # morphologies=English_Celex(legacy=True),
        morphologies=ChainedModestDatasets([English_MorphyNet_Inflections(), English_MorphyNet_Derivations()]),
        base_tokeniser=SennrichTokeniserPath(PATH_MODELBASE / "bpe-40k_oscar-en-clean"),
        reweighter=LINEAR_WEIGHTER
    )
