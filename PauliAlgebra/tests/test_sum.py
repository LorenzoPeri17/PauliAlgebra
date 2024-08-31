import pytest
import numpy as np

from PauliAlgebra import PauliVector # type: ignore
from numpy.testing import assert_allclose

@pytest.mark.parametrize('v1, v2', [(np.random.random(4), np.random.random(4)) for _ in range(1000)])
def test_sum(v1, v2):

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = pv1.to_matrix(dtype = np.cdouble) + pv2.to_matrix(dtype = np.cdouble)

    res = (pv1 + pv2).to_matrix(dtype = np.cdouble)

    assert_allclose(res, expected)

@pytest.mark.parametrize('v1r, v1i, v2r, v2i', [(np.random.random(4), np.random.random(4), np.random.random(4), np.random.random(4)) for _ in range(1000)])
def test_sum_complex(v1r, v1i, v2r, v2i):

    v1 = v1r + 1j * v1i
    v2 = v2r + 1j * v2i

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = pv1.to_matrix(dtype = np.cdouble) + pv2.to_matrix(dtype = np.cdouble)

    res = (pv1 + pv2).to_matrix(dtype = np.cdouble)

    assert_allclose(res, expected)

@pytest.mark.parametrize('v1, v2', [(np.random.random(4), np.random.random(4)) for _ in range(1000)])
def test_sub(v1, v2):

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = pv1.to_matrix(dtype = np.cdouble) - pv2.to_matrix(dtype = np.cdouble)

    res = (pv1 - pv2).to_matrix(dtype = np.cdouble)

    assert_allclose(res, expected)

@pytest.mark.parametrize('v1r, v1i, v2r, v2i', [(np.random.random(4), np.random.random(4), np.random.random(4), np.random.random(4)) for _ in range(1000)])
def test_sub_complex(v1r, v1i, v2r, v2i):

    v1 = v1r + 1j * v1i
    v2 = v2r + 1j * v2i
    
    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = pv1.to_matrix(dtype = np.cdouble) - pv2.to_matrix(dtype = np.cdouble)

    res = (pv1 - pv2).to_matrix(dtype = np.cdouble)
    
    assert_allclose(res, expected)