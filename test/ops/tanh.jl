using Flimsy
using Base.Test

x = GradVariable(randn(3))
y = tanh(x)
@test size(y) == (3, 1)
@test all(tanh(x.data) .== y.data)
test_op_grad_mse(tanh, x, wrt=x)

x = GradVariable(randn(3, 5))
y = tanh(x)
@test size(y) == (3, 5)
@test all(tanh(x.data) .== y.data)
test_op_grad_mse(tanh, x, wrt=x)