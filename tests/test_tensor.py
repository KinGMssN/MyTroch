import unittest
import numpy as np
from mytorch.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_add(self):
        x = Tensor(3)
        y = Tensor(4)

        z = x + y
        z.backward()

        self.assertEqual(z.data, 7)
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, 1)

    def test_sub(self):
        x = Tensor(7)
        y = Tensor(4)

        z = x - y
        z.backward()

        self.assertEqual(z.data, 3)
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, -1)

    def test_mul(self):
        x = Tensor(3)
        y = Tensor(4)

        z = x * y
        z.backward()

        self.assertEqual(z.data, 12)
        self.assertEqual(x.grad, 4)
        self.assertEqual(y.grad, 3)

    def test_div(self):
        x = Tensor(8)
        y = Tensor(2)

        z = x / y
        z.backward()

        self.assertEqual(z.data, 4)
        self.assertEqual(x.grad, 0.5)
        self.assertEqual(y.grad, -2)

    def test_pow(self):
        x = Tensor(3)
        y = Tensor(2)

        z = x ** y
        z.backward()

        self.assertEqual(z.data, 9)
        self.assertAlmostEqual(x.grad, 6)
        self.assertAlmostEqual(y.grad, 9 * np.log(3))

    def test_chain(self):
        x = Tensor(3)

        y = x * x
        z = y * x

        z.backward()

        self.assertEqual(z.data, 27)
        self.assertEqual(x.grad, 27)

    def test_gradient_accumulation(self):
        x = Tensor(3)

        a = x * x
        b = x * x
        c = a + b

        c.backward()

        self.assertEqual(c.data, 18)
        self.assertEqual(x.grad, 12)

    def test_broadcast_add(self):
        x = Tensor([1, 2, 3])
        y = Tensor([10])

        z = x + y
        z.backward()

        np.testing.assert_array_equal(z.data, [11, 12, 13])
        np.testing.assert_array_equal(x.grad, [1, 1, 1])
        np.testing.assert_array_equal(y.grad, [3])

    def test_broadcast_mul(self):
        x = Tensor([1, 2, 3])
        y = Tensor([10])

        z = x * y
        z.backward()

        np.testing.assert_array_equal(z.data, [10, 20, 30])
        np.testing.assert_array_equal(x.grad, [10, 10, 10])
        np.testing.assert_array_equal(y.grad, [6])

    def test_complex_graph(self):
        x = Tensor(2)
        y = Tensor(3)

        a = x ** Tensor(2)
        b = Tensor(2) * y
        c = a + b
        z = c ** Tensor(2)

        z.backward()

        self.assertEqual(z.data, 100)
        self.assertEqual(x.grad, 80)
        self.assertEqual(y.grad, 40)


if __name__ == "__main__":
    unittest.main()