import pytest
import numpy as np

from PauliAlgebra.Pauli import PauliVector

def _are_close(a, b):
    return np.allclose(np.array(a, dtype = np.cdouble), np.array(b, dtype = np.cdouble))

def _commutator(a, b):
    return a @ b - b @ a

def _anticommutator(a, b):
    return a @ b + b @ a

@pytest.mark.parametrize('v1, v2', [(np.random.random(4), np.random.random(4)) for _ in range(2500)])
def test_commutator(v1, v2):

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = _commutator(pv1.to_matrix(dtype = np.cdouble), pv2.to_matrix(dtype = np.cdouble))

    res = (PauliVector.commutator(pv1, pv2)).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)

@pytest.mark.parametrize('v1r, v1i, v2r, v2i', [(np.random.random(4), np.random.random(4), np.random.random(4), np.random.random(4)) for _ in range(2500)])
def test_commutator_complex(v1r, v1i, v2r, v2i):

    v1 = v1r + 1j * v1i
    v2 = v2r + 1j * v2i

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = _commutator(pv1.to_matrix(dtype = np.cdouble), pv2.to_matrix(dtype = np.cdouble))

    res = (PauliVector.commutator(pv1, pv2)).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)

@pytest.mark.parametrize('v1, v2', [(np.random.random(4), np.random.random(4)) for _ in range(2500)])
def test_anticommutator(v1, v2):

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = _anticommutator(pv1.to_matrix(dtype = np.cdouble), pv2.to_matrix(dtype = np.cdouble))

    res = (PauliVector.anticommutator(pv1, pv2)).to_matrix(dtype = np.cdouble)

    print(expected)
    print(res)

    assert _are_close(res, expected)

@pytest.mark.parametrize('v1r, v1i, v2r, v2i', [(np.random.random(4), np.random.random(4), np.random.random(4), np.random.random(4)) for _ in range(2500)])
def test_anticommutator_complex(v1r, v1i, v2r, v2i):

    v1 = v1r + 1j * v1i
    v2 = v2r + 1j * v2i

    pv1 = PauliVector(v1)
    pv2 = PauliVector(v2)

    expected = _anticommutator(pv1.to_matrix(dtype = np.cdouble), pv2.to_matrix(dtype = np.cdouble))

    res = (PauliVector.anticommutator(pv1, pv2)).to_matrix(dtype = np.cdouble)

    assert _are_close(res, expected)