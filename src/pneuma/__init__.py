import logging

# Suppress warnings related to non-CUDA JAX usage by bm25s
logger = logging.getLogger("jax._src.xla_bridge")
logger.setLevel(logging.ERROR)

from pneuma.pneuma import Pneuma
