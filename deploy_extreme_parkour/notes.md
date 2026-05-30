# Heightmap policy deployment

This directory contains the non-Isaac runtime utilities for the heightmap student policy.

The runtime input layout is:

- 49 proprioception values without contact flags
- 132 local terrain-height samples
- 10 frames of 53-value proprioception history, with the four contact positions set to zero

The exported policy keeps the original actor layout and injects the 32-value terrain latent produced by the heightmap encoder.
