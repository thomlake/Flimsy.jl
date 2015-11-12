using Flimsy
using Base.Test


# Constant - MxN
c = Float64(pi)
x = Variable(randn(5, 3))
y = minus(c, x)
@test size(y) == (5, 3)
@test all(y.data .== c .- x.data)
x = Variable(randn(5, 3))
test_op_grad((s)->minus(s, c, x), ()->minus(c, x), x)

# MxN - MxN
x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
y = minus(x1, x2)
@test size(y) == (5, 3)
@test all(y.data .== x1.data .- x2.data)

x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
test_op_grad((s)->minus(s, x1, x2), ()->minus(x1, x2), x1)
x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
test_op_grad((s)->minus(s, x1, x2), ()->minus(x1, x2), x2)
