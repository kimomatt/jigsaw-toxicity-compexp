"""Dataset-agnostic concept annotation package."""
# runs every time someone imports the concepts package, so we want to keep it lightweight and not do any heavy computations here. We just want to expose the main classes and functions that users will interact with, and then they can import the specific modules for the different tiers if they want to use those concepts or build their own.

from .base import Concept, ConceptSet
from .build import build_concept_set
from .tier1_words import build_tier1_vocabulary, make_word_concepts

# if someone does from concepts import *, they will get these names in their namespace, but not the tier2_primitives or tier3_clusters modules themselves, which are more for internal use within the package
__all__ = [
    "Concept",
    "ConceptSet",
    "build_concept_set",
    "build_tier1_vocabulary",
    "make_word_concepts",
]
