def test_chained_backward():
    x = Tensor(3)
    y = Tensor(4)

    a = x * y
    b = a + x

    b.backward()

    assert b.data == 15
    assert x.grad == 5
    assert y.grad == 3