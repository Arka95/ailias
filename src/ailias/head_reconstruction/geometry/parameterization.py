"""Shared physical parameter bounds for CPU and batched CUDA fitting."""

import math

import numpy as np


def parameter_bounds(policy, view_count, expression_count, identity_count):
    identity_limit = policy.identity_coefficient_limit
    expression_limit = policy.expression_coefficient_limit
    lower = np.r_[np.full(identity_count, -identity_limit), np.tile(np.r_[
        [-7]*3, [-5, -5, .06], math.log(policy.focal_factor_min),
        [-expression_limit]*expression_count], view_count)]
    upper = np.r_[np.full(identity_count, identity_limit), np.tile(np.r_[
        [7]*3, [5, 5, 10], math.log(policy.focal_factor_max),
        [expression_limit]*expression_count], view_count)]
    return lower, upper
