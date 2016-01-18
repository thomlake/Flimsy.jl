using Flimsy
using Base.Test

x = GradVariable(randn(3))
y = relu(x)
@test size(y) == (3, 1)
@test all(relu(x.data) .== y.data)
test_op_grad_mse(relu, x, wrt=x)

x = GradVariable(randn(3, 5))
y = relu(x)
@test size(y) == (3, 5)
@test all(relu(x.data) .== y.data)
test_op_grad_mse(relu, x, wrt=x)