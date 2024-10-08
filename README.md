# PauliAlgebra

![win](https://github.com/LorenzoPeri17/PauliAlgebra/actions/workflows/Windows.yaml/badge.svg)
![ubu](https://github.com/LorenzoPeri17/PauliAlgebra/actions/workflows/Ubuntu.yaml/badge.svg)
![mac](https://github.com/LorenzoPeri17/PauliAlgebra/actions/workflows/macOs.yaml/badge.svg)

`PauliAlgebra` is a module to deal with **exact** calculations of Pauli matrices

The interface to the module is the class `PauliVector` that can be used to instantiate any `2x2` Hermitian matrix

```python
from PauliAlgebra import PauliVector

M = PauliVector([
        1, # Identity
        2, # sigma x
        0, # sigma y
        1j # sigma z
    ])
```

> `PauliVector` is fully compatible with `sympy` expressions!

Alternatively, the module exposes the standard matrices

* `Id` : `2x2` Identity
* `sigma_x`
* `sigma_y`
* `sigma_z`
* `sigma_plus`
  
    ``` python
    sigma_plus = (sigma_x + 1j*sigma_y)/2 
    # ((0,1)
    #  (0,0))
    ```

* `sigma_minus` = `(sigma_x - 1j*sigma_y)/2`
  
    ``` python
    sigma_minus = (sigma_x - 1j*sigma_y)/2 
    # ((0,0)
    #  (1,0))
    ```

* `P_up`
  
    ``` python
    P_up = (Id + sigma_z)/2
    # ((1,0)
    #  (0,0))
    ```

* `P_down`
  
    ``` python
    P_down = (Id - sigma_z)/2
    # ((0,0)
    #  (0,1))
    ```

## Arithmetic Operations

`PauliVector` supports the following arithmetic operations:

* Addition and subtraction with another `PauliVector`
* Multiplication with a scalar or another `PauliVector` (performs matrix multiplication)
* Division by a scalar

So the above example could have been written as

```python
from PauliAlgebra import (
    Id,
    sigma_x,
    sigma_z
)

M = Id + 2*sigma_x + 1j*sigma_z
```

## Commutators and Anticommutators

This module allows for fast adn exact computation of commutators and anticommutators of two `PauliVector` using the relationship

$$
\left(\vec{a} \cdot \vec{\sigma}\right)\left(\vec{b} \cdot \vec{\sigma}\right) = Id~ \left(\vec{a} \cdot\vec{b} \right) + i \left(\vec{a} \times\vec{b} \right) \cdot \vec{\sigma}
$$

and the (anti)commutativity of dot and cross product.

```python
A = sigma_x
B = sigma_y

commAB = PauliVector.commutator(A,B) # = 2j*sigma_z
anticommAB = PauliVector.anticommutator(A,B) # = 0
```

## Exponentiation

`PauliVector` supports exponentiation with the standard formula

$$
\exp\left(i \theta \hat{n} \cdot \vec{\sigma} \right) = Id~ \cos{\theta} + i \hat{n} \cdot \vec{\sigma} \sin{\theta}
$$

```python
M = -1j*np.pi*sigma_x

expM = M.exponentiate() # == Id
```

## Usage with `sympy` expressions

`PauliVector` is fully compatible with `sympy` expressions!

```python
theta = sp.symbols(r'\theta')

M = (Id* sp.sin(theta) + sigma_z*sp.cos(theta))/sp.sqrt(2)

M.to_sp().applyfunc(sp.trigsimp)
# [sqrt(2)*sin(\theta + pi/4),                           0],
# [                         0, -sqrt(2)*cos(\theta + pi/4)]]
```

> For complex expressions `M.simplify()` will simplify the `Id`, `x`,`y`, and `z` components

## Going back to `numpy` or `sympy`

Once you are done performing algebra on a `PauliVector` you can turn it back into more common types

* `M.toMatrix()` -> `np.ndarray` (`shape==(2,2)`)
* `M.to_sp()` -> `sp.Matrix` (`shape==(2,2)`)

## Installation

`PauliAlgebra` is available on pypi!
It can be yours by simply

```bash
$ pip install PauliAlgebra
```

in the environment of your choice!
