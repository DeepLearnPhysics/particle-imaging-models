"""Triton kernels behind the fused paths of pimm.ops; import them through the wrappers in pimm.ops.

The exceptions are ``fps``, ``ball_query``, ``knn_points`` and ``cnms``, imported directly by
``pimm.models.polarmae.ops`` for CUDA tensors.
"""
