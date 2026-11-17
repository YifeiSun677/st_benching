# Python >= 3.10 removed the ABC aliases from `collections` (they live in `collections.abc`).
# The upstream Hist2ST repo does `from easydl import *`, and easydl still does
# `from collections import Iterable`, so restore the aliases before anything imports it.
import collections
import collections.abc

for _n in ("Iterable", "Iterator", "Mapping", "MutableMapping", "Sequence", "MutableSequence",
           "Set", "MutableSet", "Callable", "Hashable", "Sized", "Container", "Generator"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))
