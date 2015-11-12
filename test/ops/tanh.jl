using Flimsy
using Base.Test

x = Variable(randn(3))
y = tanh(x)
@test size(y) == (3, 1)
@test all(tanh(x.data) .== y.data)
test_op_grad((s)->tanh(s, x), ()->tanh(x), x)

x = Variable(randn(3, 5))
y = tanh(x)
@test size(y) == (3, 5)
@test all(tanh(x.data) .== y.data)
test_op_grad((s)->tanh(s, x), ()->tanh(x), x)
