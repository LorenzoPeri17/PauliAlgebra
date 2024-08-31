import numpy as np
from typing import Union, Literal
from numbers import Number
from sympy import S
import sympy as sp

def epsilon(i,j,k):
    return (i-j)*(j-k)*(k-i)/2

class PauliVector():

    _Id = np.eye(2, dtype = object)

    _x = np.array([[0,1],
                   [1,0]],dtype=object)
    
    _y = np.array([[0, -1j], 
                    [1j, 0]], dtype = object)
    
    _z = np.array([[1, 0], 
                   [0, -1]], dtype = object)
    
    _Pauli_num = [_Id, _x, _y, _z]

    def __init__(self,vec : np.ndarray[object, Literal[4]]):
        self.vec = vec

    def __repr__(self) -> str:
        return f'(Id:{self.vec[0]}, X:{self.vec[1]}, Y:{self.vec[2]}, Z:{self.vec[3]})'
    
    @property
    def Id(self) -> object:
        return self.vec[0]
    
    @property
    def trace(self) -> object:
        return self.vec[0]/2
    
    @property
    def x(self) -> object:
        return self.vec[1]
    
    @property
    def y(self) -> object:
        return self.vec[2]
    
    @property
    def z(self) -> object:
        return self.vec[3]

    def __eq__(self, other : 'PauliVector') -> bool:
        try:
            return np.allclose(self.vec,other.vec)
        except Exception:
            return all([sp.simplify(self.vec[i] - other.vec[i]) == 0 for i in range(4)])
    
    def __add__(self,other : 'PauliVector') -> 'PauliVector':

        if isinstance(other,PauliVector):
            return PauliVector(self.vec + other.vec)
    
    def __radd__(self,other : 'PauliVector') -> 'PauliVector':

        if isinstance(other,PauliVector):
            return PauliVector(self.vec + other.vec)
        
    def __sub__(self,other : 'PauliVector') -> 'PauliVector':

        if isinstance(other,PauliVector):
            return PauliVector(self.vec - other.vec)
    
    def __rsub__(self,other : 'PauliVector') -> 'PauliVector':

        if isinstance(other,PauliVector):
            return PauliVector(self.vec - other.vec)
        
    def __mul__(self,other : Union['PauliVector', Number, S]) -> 'PauliVector':
        if isinstance(other,PauliVector):
            res = np.zeros(4,dtype=object)

            res[0] = np.dot(self.vec,other.vec)

            for k, v in enumerate(self.cross(self.vec[1:],other.vec[1:])):
                res[k+1] = 1j*v + self.vec[0]*other.vec[k+1] + other.vec[0]*self.vec[k+1]
            
            return PauliVector(res)
        else:
            return PauliVector(self.vec * other)
        
    def __rmul__(self,other : Union['PauliVector', Number, S]) -> 'PauliVector':
        if isinstance(other,PauliVector):
            res = np.zeros(4,dtype=object)

            res[0] = np.dot(self.vec,other.vec)

            for k, v in enumerate(self.cross(other.vec[1:],self.vec[1:])):
                res[k+1] = 1j*v + self.vec[0]*other.vec[k+1] + other.vec[0]*self.vec[k+1]
            
            return PauliVector(res)
        else:
            return PauliVector(self.vec * other)
        
    def __truediv__(self,other : Number) -> 'PauliVector':
        return PauliVector(self.vec / other)
    
    def to_matrix(self, dtype = object) -> np.ndarray[Union[complex,object], (2, 2)]:
        return np.sum((self.vec[i]*PauliVector._Pauli_num[i] for i in range(4)), dtype = dtype)
    
    def to_sp(self)-> sp.Matrix:
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

            argument = sp.simplify(-1j * norm)
            _reduced_vec = [sp.simplify(1j*sp.sin(argument)*v) for v in versor]
            exp_vec = np.array([sp.simplify(sp.cos(argument))] + _reduced_vec,dtype=object)
            
            return PauliVector(sp.exp(self.vec[0])*exp_vec)
        
    def adjoint(self) -> 'PauliVector':
        return PauliVector(np.conj(self.vec))
    
    @staticmethod
    def cross(a : np.ndarray[object, Literal[3]], b : np.ndarray[object, Literal[3]]) -> np.ndarray[object, Literal[3]]:
        return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]],dtype=object)
    
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

if __name__ == '__main__':
    print(PauliVector.commutator(sigma_x,sigma_y))
    print(sigma_x*sigma_y)