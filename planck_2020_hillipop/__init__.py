from ._version import get_versions
from .hillipop import EE, TE, TT, TTTE, TTTEEE

__version__ = get_versions()["version"]
del get_versions
