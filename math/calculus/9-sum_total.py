#!/usr/bin/env python3
"""Our life is the sum total of all the decisions we make every day,
and those decisions are determined by our priorities"""


def summation_i_squared(n):
    """ This calculates sum{i=1}^{n} i^2:"""
    if type(n) is not int or n is None or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
