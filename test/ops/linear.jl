using Flimsy
using Base.Test

x = GradVariable(randn(3))
w = GradVariable(randn(4, 3))
y = linear(w, x)
@test size(y) == (4, 1)
@test all((w.data * x.data) .== y.data)
test_op_grad_mse(linear, w, x, wrt=[w, x])

x = GradVariable(randn(3, 5))
w = GradVariable(randn(4, 3))
y = linear(w, x)
@test size(y) == (4, 5)
@test all((w.data * x.data) .== y.data)
test_op_grad_mse(linear, w, x, wrt=[w, x])