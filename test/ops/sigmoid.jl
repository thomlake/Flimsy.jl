using Flimsy
using Base.Test

x = GradVariable(randn(3))
y = sigmoid(x)
@test size(y) == (3, 1)
@test all(sigmoid(x.data) .== y.data)
test_op_grad_mse(sigmoid, x, wrt=x)

x = GradVariable(randn(3, 5))
y = sigmoid(x)
@test size(y) == (3, 5)
@test all(sigmoid(x.data) .== y.data)
test_op_grad_mse(tanh, x, wrt=x)