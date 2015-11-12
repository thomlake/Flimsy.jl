using Flimsy
using Base.Test

x = Variable(randn(3))
y = relu(x)
@test size(y) == (3, 1)
@test all(max(0, x.data) .== y.data)
test_op_grad((s)->relu(s, x), ()->relu(x), x)

x = Variable(randn(3, 5))
y = relu(x)
@test size(y) == (3, 5)
@test all(max(0, x.data) .== y.data)
test_op_grad((s)->relu(s, x), ()->relu(x), x)
