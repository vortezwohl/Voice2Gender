"""Public package interface for the bundled voice gender classifier.

The package exports the high-level prediction function and keeps model loading
implementation details in private modules.
"""

from ._predict import predict

__AUTHOR__ = "吴子豪"
__EMAIL__ = "vortez.wohl@gmail.com"
