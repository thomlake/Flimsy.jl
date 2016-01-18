using Flimsy
using Base.Test


m, n = 6, 1
x = GradVariable(randn(m))
y = identity(x)
@test all(x.data .== y.data)
test_op_grad_mse(identity, x, wrt=x)

m, n = 3, 9
x = GradVariable(randn(m, n))
y = identity(x)
@test all(x.data .== y.data)
test_op_grad_mse(identity, x, wrt=x)
