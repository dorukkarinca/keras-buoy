# -*- coding: utf-8 -*-

import pytest
from keras_buoy.skeleton import fib

__author__ = "Doruk Karınca"
__copyright__ = "Doruk Karınca"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
