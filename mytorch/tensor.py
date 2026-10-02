import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return "Tensor(" + str(self.data) + ")"

    def __add__(self, other):
        out = Tensor(self.data + other.data)
        out._prev =  {self, other}

        def _backward():
            if self.grad is None:
                self.grad = out.grad
            else:
                self.grad = self.grad + out.grad

            if other.grad is None:
                other.grad = out.grad
            else:
                other.grad = other.grad + out.grad

        out._backward = _backward

        return out


    def __sub__(self, other):
        out = Tensor(self.data - other.data)
        out._prev = {self, other}

        def _backward():
            if self.grad is None:
                self.grad = out.grad
            else :
                self.grad = self.grad + out.grad

            if other.grad is None:
                other.grad = -out.grad
            else:
                other.grad = other.grad - out.grad

        out._backward = _backward
        return out



    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            if self.grad is None:
                self.grad = other.data*out.grad
            else:
                self.grad = self.grad + other.data*out.grad

            if other.grad is None:
                other.grad = self.data*out.grad
            else :
                other.grad = other.grad + self.data*out.grad

        out._backward = _backward
        return out


    def __truediv__(self, other):
        out = Tensor(self.data / other.data)
        out._prev = {self, other}

        def _backward():
            if self.grad is None:
                self.grad = out.grad*(1/other.data)
            else :
                self.grad = self.grad + out.grad*(1/other.data)

            if other.grad is None:
                other.grad = -out.grad*self.data*(1/(other.data*other.data))
            else :
                other.grad = other.grad - out.grad*self.data*(1/(other.data*other.data))

        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Tensor(self.data ** other.data)
        out._prev = {self, other}

        def _backward():
            if self.grad is None:
                self.grad = out.grad*other.data*(self.data**(other.data-1))
            else:
                self.grad = self.grad + out.grad*other.data*(self.data**(other.data-1))

            if other.grad is None:
                other.grad = out.grad*(self.data**other.data)*np.log(self.data)
            else :
                other.grad = other.grad  + out.grad*(self.data**other.data)*np.log(self.data)

        out._backward = _backward
        return out

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