import numpy as np
from numpy.ma.core import outer


class Tensor:
    #Tesnor Basic props
    def __init__(self, data):
        self.data = np.array(data)
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return "Tensor(" + str(self.data) + ")"



    def __add__(self, other):

        other = self._ensure_tensor(other)

        out = Tensor(self.data + other.data)
        out._prev =  {self, other}

        def _backward():
            self_grad = self._unbroadcast(out.grad , self.data.shape)
            other_grad = self._unbroadcast(out.grad, other.data.shape)
            if self.grad is None:
                self.grad = self_grad
            else:
                self.grad = self.grad + self_grad

            if other.grad is None:
                other.grad = other_grad
            else:
                other.grad = other.grad + other_grad

        out._backward = _backward

        return out


    def __sub__(self, other):

        other = self._ensure_tensor(other)

        out = Tensor(self.data - other.data)
        out._prev = {self, other}

        def _backward():
            self_grad = self._unbroadcast(out.grad, self.data.shape)
            other_grad = self._unbroadcast(-out.grad, other.data.shape)
            if self.grad is None:
                self.grad = self_grad
            else :
                self.grad = self.grad + self_grad

            if other.grad is None:
                other.grad = other_grad
            else:
                other.grad = other.grad + other_grad

        out._backward = _backward
        return out



    def __mul__(self, other):

        other = self._ensure_tensor(other)

        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self_grad = self._unbroadcast(other.data * out.grad, self.data.shape)
            other_grad = self._unbroadcast(self.data * out.grad, other.data.shape)

            if self.grad is None:
                self.grad = self_grad
            else:
                self.grad = self.grad + self_grad

            if other.grad is None:
                other.grad = other_grad
            else :
                other.grad = other.grad +other_grad

        out._backward = _backward
        return out


    def __truediv__(self, other):

        other = self._ensure_tensor(other)

        out = Tensor(self.data / other.data)
        out._prev = {self, other}

        def _backward():
            self_grad = self._unbroadcast(out.grad*(1/other.data), self.data.shape)
            other_grad = self._unbroadcast(-out.grad*self.data*(1/(other.data*other.data)), other.data.shape)

            if self.grad is None:
                self.grad = self_grad
            else :
                self.grad = self.grad + self_grad

            if other.grad is None:
                other.grad = other_grad
            else :
                other.grad = other.grad + other_grad

        out._backward = _backward
        return out

    def __pow__(self, other):

        other = self._ensure_tensor(other)

        out = Tensor(self.data ** other.data)
        out._prev = {self, other}

        def _backward():
            self_grad = self._unbroadcast(out.grad*other.data*(self.data**(other.data-1)), self.data.shape)
            other_grad = self._unbroadcast(out.grad*(self.data**other.data)*np.log(self.data), other.data.shape)
            if self.grad is None:
                self.grad = self_grad
            else:
                self.grad = self_grad

            if other.grad is None:
                other.grad = other_grad
            else :
                other.grad = other_grad

        out._backward = _backward
        return out
    #Basic Arithmetic
    def neg(self):
        out = Tensor(-self.data)
        out._prev = {self}

        def _backward():
            grad = -out.grad

            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def __abs__(self):
        out = Tensor(np.abs(self.data))
        out._prev = {self}

        def _backward():
            grad = np.sign(self.grad)*out.grad
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def sign(self):
        out = Tensor(np.abs(self.data))
        out._prev = {self}

        def _backward():
            grad = np.zeros_like(self.data)
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def reciprocal(self):
        out = Tensor(np.reciprocal(self.data))
        out._prev = {self}

        def _backward():
            grad = -out.grad/(self.data**2)
            if self.grad is None:
                self.grad = grad
            else :
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def square(self):
        out = Tensor(np.square(self.data))
        out._prev = {self}

        def _backward():
            grad = 2*out.grad*self.data

            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad
        out._backward = _backward

        return out

    def cube(self):
        out = Tensor(np.power(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad*3*np.square(self.data)
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad/(2*out.data)
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad
        out._backward = _backward

        return out

    def cbrt(self):
        out = Tensor(np.cbrt(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad/(3*(self.data**2/3))
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad
        out._backward = _backward

        return out

    #exponential operations
    def exp(self):
        out = Tensor(np.exp(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad*out.data
            if self.grad is None:
                self.grad = grad
            else :
                self.grad = self.grad + grad

        out._backward = _backward

        return out
    def log(self):
        out = Tensor(np.log(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad/self.data
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def log2(self):
        out = Tensor(np.log2(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad / (self.data*np.log())
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def log10(self):
        out = Tensor(np.log10(self.data))
        out._prev = {self}
        def _backward():
            grad = out.grad/(self.data*np.log(10))
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    #Trigonometric operations
    def sin(self):
        out = Tensor(np.sin(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad*np.cos(self.data)
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        return out
    def cos(self):
        out = Tensor(np.cos(self.data))
        out._prev = {self}

        def _backward():
            grad = -out.grad*np.sin(self.data)
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def tan(self):
        out = Tensor(np.tan(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad/np.square(np.cos(self.data))
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def cot(self):
        out = Tensor(1/np.tan(self.data))
        out._prev = {self}

        def _backward():
            grad = -out.grad/np.square(np.sin(self.data))
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def sec(self):
        out = Tensor(1/np.cos(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad*(np.sin(self.data))/(np.square(np.cos(self.data)))
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def cosec(self):
        out = Tensor(1/np.sin(self.data))
        out._prev = {self}

        def _backward():
            grad = -out.grad*(np.cos(self.data))/(np.square(np.sin(self.data)))
            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    #Inverse Trigonometric operations
    def arcsin(self):
        out = Tensor(np.arcsin(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad / np.sqrt(1 - self.data ** 2)

            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def arccos(self):
        out = Tensor(np.arccos(self.data))
        out._prev = {self}

        def _backward():
            grad = -out.grad / np.sqrt(1 - self.data ** 2)

            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def arctan(self):
        out = Tensor(np.arctan(self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad / (1 + self.data ** 2)

            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def arccot(self):
        out = Tensor(np.pi / 2 - np.arctan(self.data))
        out._prev = {self}

        def _backward():
            grad = -out.grad / (1 + self.data ** 2)

            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def arcsec(self):
        out = Tensor(np.arccos(1 / self.data))
        out._prev = {self}

        def _backward():
            grad = out.grad / (np.abs(self.data) * np.sqrt(self.data ** 2 - 1))

            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    def arccosec(self):
        out = Tensor(np.arcsin(1 / self.data))
        out._prev = {self}

        def _backward():
            grad = -out.grad / (
                    np.abs(self.data) * np.sqrt(self.data ** 2 - 1)
            )

            if self.grad is None:
                self.grad = grad
            else:
                self.grad = self.grad + grad

        out._backward = _backward

        return out

    #Reverse Operations

    def backward(self):
        # Topological sort/order of the computation graph

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)

                # pt means previous tensor
                for pt in tensor._prev:
                    build_topo(pt)

                topo.append(tensor)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for tensor in reversed(topo):
            tensor._backward()

    def _unbroadcast(self, grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis in range(len(shape)):
            size = shape[axis]

            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad



    #Helper Functions
    def _ensure_tensor(self,other):
        if not isinstance(other, Tensor):
            Other = Tensor(other)
        return other

    def __radd__(self,other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other - self

    def __rtruediv__(self, other):
        return other / self

    def __rpow__(self, other):
        other = self._ensure_tensor(other)
        return other ** self

