using Flimsy
using Base.Test

# mult(Mx1, Mx1)
m, n = 3, 1
a = GradVariable(randn(m))
b = GradVariable(randn(m))
c = mult(a, b)
@test size(c) == (m, n)
@test all(a.data .* b.data .== c.data)
test_op_grad_mse(mult, a, b, wrt=[a, b])

# mult(MxN, MxN)
m, n = 5, 12
a = GradVariable(randn(m, n))
b = GradVariable(randn(m, n))
c = mult(a, b)
@test size(c) == (m, n)
@test all(a.data .* b.data .== c.data)
test_op_grad_mse(mult, a, b, wrt=[a, b])

# mult(1x1, Mx1)
m, n = 4, 1
a = GradVariable(randn())
b = GradVariable(randn(m))
c = mult(a, b)
@test size(c) == (m, n)
@test all(a.data .* b.data .== c.data)
test_op_grad_mse(mult, a, b, wrt=[a, b])

# mult(1xN, MxN)
m, n = 4, 7
a = GradVariable(randn(1, n))
b = GradVariable(randn(m, n))
c = mult(a, b)
@test size(c) == (m, n)
@test all(a.data .* b.data .== c.data)
test_op_grad_mse(mult, a, b, wrt=[a, b])