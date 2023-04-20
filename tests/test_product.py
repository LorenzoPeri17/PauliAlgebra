import pytest
import numpy as np

from PauliAlgebra.Pauli import PauliVector

def _are_close(a, b):
    return np.allclose(np.array(a, dtype = np.cdouble), np.array(b, dtype = np.cdouble))

@pytest.mark.parametrize('v1, v2', [(np.random.random(4), np.random.random(4)) for _ in range(1000)])
def test_product(v1, v2):

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = pv1.to_matrix(dtype = np.cdouble) @ pv2.to_matrix(dtype = np.cdouble)

    res = (pv1 * pv2).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)

@pytest.mark.parametrize('v1r, v1i, v2r, v2i', [(np.random.random(4), np.random.random(4), np.random.random(4), np.random.random(4)) for _ in range(1000)])
def test_product_complex(v1r, v1i, v2r, v2i):

    v1 = v1r + 1j * v1i
    v2 = v2r + 1j * v2i

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = pv1.to_matrix(dtype = np.cdouble) @ pv2.to_matrix(dtype = np.cdouble)

    res = (pv1 * pv2).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)

@pytest.mark.parametrize('v1, x', [(np.random.random(4), np.random.random()) for _ in range(1000)])
def test_product_scalar(v1, x):

    pv1 = PauliVector(v1)

    expected = x * pv1.to_matrix(dtype = np.cdouble)

    res = (x * pv1).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)

    res = (pv1 * x).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)

@pytest.mark.parametrize('v1r, v1i, x, y', [(np.random.random(4), np.random.random(4), np.random.random(), np.random.random()) for _ in range(1000)])
def test_product_complex_scalar(v1r, v1i, x, y):

    v1 = v1r + 1j * v1i
    z = x + 1j * y

    pv1 = PauliVector(v1)
    
    expected = z * pv1.to_matrix(dtype = np.cdouble)

    res = (z * pv1).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)

    res = (pv1 * z).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)