import pytest
import numpy as np
from numpy.testing import assert_allclose

from PauliAlgebra import PauliVector # type: ignore

_Id = np.eye(2, dtype = np.cdouble)

_x = np.array([[0,1],
                [1,0]], dtype = np.cdouble)

_y = np.array([[0, -1j], 
                [1j, 0]], dtype = np.cdouble)

_z = np.array([[1, 0], 
                [0, -1]], dtype = np.cdouble)

_Pauli_num = [_Id, _x, _y, _z]

@pytest.mark.parametrize('v', [np.random.random(4) for _ in range(1000)])
def test_expansion(v):

    pv = PauliVector(v)

    expected = np.zeros((2,2), dtype = np.cdouble)

    for i, vv in enumerate(v):
        expected += vv * _Pauli_num[i]
    
    res = pv.to_matrix(dtype = np.cdouble)
    
    assert_allclose(res, expected)

@pytest.mark.parametrize('vr, vi', [(np.random.random(4), np.random.random(4)) for _ in range(1000)])
def test_expansion_complex(vr, vi):

    v = vr + 1j * vi

    pv = PauliVector(v)

    expected = np.zeros((2,2), dtype = np.cdouble)

    for i, vv in enumerate(v):
        expected += vv * _Pauli_num[i]
        
    expected = np.array(expected, dtype=np.cdouble)
    
    res = pv.to_matrix(dtype = np.cdouble)

    assert_allclose(res, expected)

@pytest.mark.parametrize('v1, v2', [(np.random.random(3), np.random.random(3)) for _ in range(1000)])
def test_cross(v1, v2):

    res = PauliVector.cross(v1, v2)

    expected = np.cross(v1, v2)

    assert_allclose(np.array(res, dtype=np.double), expected)


@pytest.mark.parametrize('v1r, v1i, v2r, v2i', [(np.random.random(3), np.random.random(3), np.random.random(3), np.random.random(3)) for _ in range(1000)])
def test_cross_complex(v1r, v1i, v2r, v2i):

    v1 = v1r + 1j * v1i
    v2 = v2r + 1j * v2i

    res = PauliVector.cross(v1, v2)

    expected = np.cross(v1, v2)

    assert_allclose(np.array(res, dtype=np.cdouble), expected)