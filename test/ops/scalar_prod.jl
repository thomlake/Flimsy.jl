using Flimsy
using Base.Test

m, n = 3, 1
a = 0.5
x = GradVariable(randn(m))
y = prod(a, x)
@test size(y) == (m, n)
@test all(a .* x.data .== y.data)
test_op_grad_mse(prod, a, x, wrt=x)

a = -0.5
x = GradVariable(randn(m))
y = prod(a, x)
@test size(y) == (m, n)
@test all(a .* x.data .== y.data)
test_op_grad_mse(prod, a, x, wrt=x)

m, n = 3, 5
a = 0.5
x = GradVariable(randn(m, n))
y = prod(x, a)
@test size(y) == (m, n)
@test all(a .* x.data .== y.data)
test_op_grad_mse(prod, x, a, wrt=x)

a = -0.5
x = GradVariable(randn(m, n))
y = prod(x, a)
@test size(y) == (m, n)
@test all(a .* x.data .== y.data)
test_op_grad_mse(prod, x, a, wrt=x)
