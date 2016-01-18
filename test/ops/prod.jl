using Flimsy
using Base.Test

# prod(Mx1, Mx1)
m, n = 3, 1
a = GradVariable(randn(m))
b = GradVariable(randn(m))
c = prod(a, b)
@test size(c) == (m, n)
@test all(a.data .* b.data .== c.data)
test_op_grad_mse(prod, a, b, wrt=[a, b])

# prod(MxN, MxN)
m, n = 5, 12
a = GradVariable(randn(m, n))
b = GradVariable(randn(m, n))
c = prod(a, b)
@test size(c) == (m, n)
@test all(a.data .* b.data .== c.data)
test_op_grad_mse(prod, a, b, wrt=[a, b])

# prod(1x1, Mx1)
m, n = 4, 1
a = GradVariable(randn())
b = GradVariable(randn(m))
c = prod(a, b)
@test size(c) == (m, n)
@test all(a.data .* b.data .== c.data)
test_op_grad_mse(prod, a, b, wrt=[a, b])

# prod(1xN, MxN)
m, n = 4, 7
a = GradVariable(randn(1, n))
b = GradVariable(randn(m, n))
c = prod(a, b)
@test size(c) == (m, n)
@test all(a.data .* b.data .== c.data)
test_op_grad_mse(prod, a, b, wrt=[a, b])