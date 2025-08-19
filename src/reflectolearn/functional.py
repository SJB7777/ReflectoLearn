from collections.abc import Callable
from functools import reduce


def compose(*funcs: Callable):
    """Combines multiple functions from right to left.
    This means the rightmost function is executed first,
    and its result is passed as input to the next function.
    In this way, creates a single function from multiple functions

    compose(h, g, f)(x) is equivalent to h(g(f(x)))
    """
    return reduce(lambda f, g: lambda x: f(g(x)), funcs)


def pipe(*funcs: Callable):
    """Combines multiple functions from left to right.
    This means the leftmost function is executed first,
    and its result is passed as input to the next function.
    In this way, creates a single function from multiple functions

    pipe(f, g, h)(x) is equivalent to h(g(f(x)))
    """
    return reduce(lambda f, g: lambda x: g(f(x)), funcs)


def batch_indices(total_length, batch_size):
    """
    total_length 길이의 배열을 batch_size 크기로 나누어
    (start, end) 인덱스 튜플을 generator로 반환
    """
    for start in range(0, total_length, batch_size):
        end = min(start + batch_size, total_length)
        yield start, end
