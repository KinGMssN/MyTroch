def test_chained_backward():
    x = Tensor(3)
    y = Tensor(4)

    a = x * y
    b = a + x

    b.backward()

    assert b.data == 15
    assert x.grad == 5
    assert y.grad == 3

def test_unbroadcast_scalar():
    x = Tensor(10)

    grad = np.array([1, 1, 1])

    result = x._unbroadcast(grad, x.data.shape)

    assert result == 3

def test_add_broadcasting():
    x = Tensor([[1, 2, 3]])
    y = Tensor([[10, 20, 30],
                [40, 50, 60]])

    z = x + y
    z.backward()

    assert x.grad.tolist() == [[2, 2, 2]]
    assert y.grad.tolist() == [[1, 1, 1],
                               [1, 1, 1]]

def test_add_scalar_broadcasting():
    x = Tensor([1, 2, 3])
    y = Tensor(10)

    z = x + y
    z.backward()

    assert x.grad.tolist() == [1, 1, 1]
    assert y.grad == 3

def test_fake_failure():
    x = Tensor([1, 2, 3])
    y = Tensor(10)

    z = x + y
    z.backward()

    assert y.grad == 999