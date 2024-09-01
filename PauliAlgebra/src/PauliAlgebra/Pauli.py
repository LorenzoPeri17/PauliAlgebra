import numpy as np # type: ignore
from numpy.typing import NDArray
from typing import Any, Union, Literal, Annotated, Sequence
from numbers import Number
from sympy.core.singleton import SingletonRegistry # type: ignore
import sympy as sp # type: ignore

def epsilon(i : int,j :int, k :int) -> float:
    return (i-j)*(j-k)*(k-i)/2

Vec4 = Annotated[NDArray, Literal[4]]
Vec3 = Annotated[NDArray, Literal[3]]
Mat22 = Annotated[NDArray, Literal[2,2]]
spMat22 = Annotated[sp.Matrix, Literal[2,2]]

class PauliVector():

    _Id = np.eye(2, dtype = object)

    _x = np.array([[0,1],
                   [1,0]],dtype=object)
    
    _y = np.array([[0, -1j], 
                    [1j, 0]], dtype = object)
    
    _z = np.array([[1, 0], 
                   [0, -1]], dtype = object)
    
    _Pauli_num = [_Id, _x, _y, _z]

    def __init__(self, vec : Vec4):
        _do_raise = False
        
        if not hasattr(vec, '__len__'):
            _do_raise = True
            print('a')
        if len(vec) != 4:
            _do_raise=True
        
        if _do_raise:
            raise ValueError('PauliVector must be constructed with an iterable of 4 elements')
        
        self.vec = vec

    def __repr__(self) -> str:
        return f'(Id:{self.vec[0]}, X:{self.vec[1]}, Y:{self.vec[2]}, Z:{self.vec[3]})'
    
    @property
    def Id(self) -> Any:
        return self.vec[0]
    
    @property
    def trace(self) -> Any:
        return self.vec[0]/2
    
    @property
    def x(self) -> Any:
        return self.vec[1]
    
    @property
    def y(self) -> Any:
        return self.vec[2]
    
    @property
    def z(self) -> Any:
        return self.vec[3]

    def __eq__(self, other) -> bool:
        if isinstance(other, PauliVector):
            try: # try if numpy can deal with the types
                return np.allclose(self.vec,other.vec)
            except Exception: # maybe sympy can
                return all([sp.simplify(self.vec[i] - other.vec[i]) == 0 for i in range(4)])
        else:
            raise ValueError('PauliVector can only be safely compared with other instances of PauliVector')
        
    def __add__(self, other : 'PauliVector') -> 'PauliVector':

        if isinstance(other,PauliVector):
            return PauliVector(self.vec + other.vec)
        else:
            raise ValueError(f'Cannot add {type(other)} to PauliVector')
    
    def __radd__(self,other : 'PauliVector') -> 'PauliVector':

        if isinstance(other,PauliVector):
            return PauliVector(self.vec + other.vec)
        else:
            raise ValueError(f'Cannot add {type(other)} to PauliVector')
        
    def __sub__(self,other : 'PauliVector') -> 'PauliVector':

        if isinstance(other,PauliVector):
            return PauliVector(self.vec - other.vec)
        else:
            raise ValueError(f'Cannot subtract {type(other)} to PauliVector')
    
    def __rsub__(self,other : 'PauliVector') -> 'PauliVector':

        if isinstance(other,PauliVector):
            return PauliVector(self.vec - other.vec)
        else:
            raise ValueError(f'Cannot subtract {type(other)} to PauliVector')
        
    def __mul__(self,other : Union['PauliVector', Number, SingletonRegistry, float, complex]) -> 'PauliVector':
        if isinstance(other,PauliVector):
            res = np.zeros(4,dtype=object)

            res[0] = np.dot(self.vec,other.vec)

            for k, v in enumerate(self.cross(self.vec[1:],other.vec[1:])):
                res[k+1] = 1j*v + self.vec[0]*other.vec[k+1] + other.vec[0]*self.vec[k+1]
            
            return PauliVector(res)
        else:
            return PauliVector(self.vec * other)
        
    def __rmul__(self,other : Union['PauliVector', Number, SingletonRegistry, float, complex]) -> 'PauliVector':
        if isinstance(other,PauliVector):
            res = np.zeros(4,dtype=object)

            res[0] = np.dot(self.vec,other.vec)

            for k, v in enumerate(self.cross(other.vec[1:],self.vec[1:])):
                res[k+1] = 1j*v + self.vec[0]*other.vec[k+1] + other.vec[0]*self.vec[k+1]
            
            return PauliVector(res)
        else:
            return PauliVector(self.vec * other)
        
    def __truediv__(self,other : Union[Number,float,complex]) -> 'PauliVector':
        return PauliVector(self.vec / other)
    
    def to_matrix(self, dtype = object) -> Mat22:
        _matrix = np.zeros((2,2), dtype = object)
        for _v, _P in zip(self.vec, PauliVector._Pauli_num):
            _matrix = _matrix + _v*_P
        return np.array(_matrix, dtype=dtype)
    
    def to_sp(self)-> spMat22:
        return sp.Matrix(self.to_matrix())
    
    def simplify(self) -> 'PauliVector':
        self.vec = np.array([sp.simplify(i) for i in self.vec],dtype=object)
        return self
    
    def exponentiate(self) -> 'PauliVector':

        try:
            norm = np.sqrt(np.sum(self.vec[1:]**2))
            versor = self.vec[1:]/norm

            argument = -1j * norm

            exp_vec = np.array([np.cos(argument), *(1j*np.sin(argument)*versor)],dtype=object)
            
            return PauliVector(np.exp(self.vec[0])*exp_vec)
        except Exception: # if it fails, it's probably because of sympy
            norm = sp.sqrt(sp.simplify(sum(sp.simplify(self.vec[1:]**2))))
            versor = sp.simplify(self.vec[1:]/norm)

            argument = sp.simplify(-sp.I * norm)
            _reduced_vec = [sp.simplify(sp.I*sp.sin(argument)*v) for v in versor]
            exp_vec = np.array([sp.simplify(sp.cos(argument))] + _reduced_vec,dtype=object)
            
            return PauliVector(sp.exp(self.vec[0])*exp_vec)
        
    def adjoint(self) -> 'PauliVector':
        return PauliVector(np.conj(self.vec))
    
    @staticmethod
    def cross(a : Vec3, b : Vec3) -> Vec3:
        return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]], dtype=object)
    
    @classmethod
    def commutator(cls, a : 'PauliVector',b : 'PauliVector') -> 'PauliVector':
        return PauliVector(2j*np.array([0,*cls.cross(a.vec[1:], b.vec[1:])],dtype=object))
    
    @classmethod
    def anticommutator(cls, a : 'PauliVector',b : 'PauliVector') -> 'PauliVector':
        return PauliVector(np.array([2*np.dot(a.vec, b.vec), *(2*(a.vec[0]*b.vec[1:] + b.vec[0]*a.vec[1:]))],dtype=object))
    
Id = PauliVector(np.array([1,0,0,0],dtype=object))
sigma_x = PauliVector(np.array([0,1,0,0],dtype=object))
sigma_y = PauliVector(np.array([0,0,1,0],dtype=object))
sigma_z = PauliVector(np.array([0,0,0,1],dtype=object))

sigma_plus = (sigma_x + 1j*sigma_y)/2
sigma_minus = (sigma_x - 1j*sigma_y)/2

P_up = (Id + sigma_z)/2
P_down = (Id - sigma_z)/2
