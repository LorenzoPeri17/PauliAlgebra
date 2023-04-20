import pytest
import numpy as np
from scipy.linalg import expm

from PauliAlgebra.Pauli import PauliVector

def _are_close(a, b):
    return np.allclose(np.array(a, dtype = np.cdouble), np.array(b, dtype = np.cdouble))

@pytest.mark.parametrize('v1', [(np.random.random(4)) for _ in range(1000)])
def test_exponent(v1):

    pv1 = PauliVector(v1)

    expected = expm(np.array(pv1.to_matrix(dtype = np.cdouble), dtype = np.cdouble))

    res = (pv1.exponentiate()).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)

@pytest.mark.parametrize('v1r, v1i', [(np.random.random(4), np.random.random(4)) for _ in range(1000)])
def test_commutator_complex(v1r, v1i):

    v1 = v1r + 1j * v1i

    pv1 = PauliVector(v1)

    expected = expm(np.array(pv1.to_matrix(dtype = np.cdouble), dtype = np.cdouble))

    res = (pv1.exponentiate()).to_matrix(dtype = np.cdouble)

    print(res)
    print(expected)

    assert _are_close(res, expected)