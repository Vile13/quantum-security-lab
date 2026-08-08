"""Shared building blocks used by more than one module in this lab.

Only pieces that several modules genuinely need live here: the dataset, the
variational classifier, and the noise models. Everything specific to a single
experiment -- its sweep, its attacks, its plots -- stays in that module's own
``src/``, so a module can be read on its own and its results traced to code
that is not shared with anything else.

The rule for adding something here is that a second module must already need
it. Code moved in anticipation of reuse tends to acquire parameters for
situations that never arrive.
"""
