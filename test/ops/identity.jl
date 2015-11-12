using Flimsy
using Base.Test

x = Variable(randn(3))
y = identity(x)
@test size(y) == (3, 1)
@test all(x.data .== y.data)
test_op_grad((s)->identity(s, x), ()->identity(x), x)

x = Variable(randn(3, 5))
y = identity(x)
@test size(y) == (3, 5)
@test all(x.data .== y.data)
test_op_grad((s)->identity(s, x), ()->identity(x), x)
