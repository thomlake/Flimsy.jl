using Flimsy
using Base.Test

# MxN + MxN
x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
y = sum(x1, x2)
@test size(y) == (5, 3)
@test all(y.data .== x1.data .+ x2.data)

x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
test_op_grad((s)->sum(s, x1, x2), ()->sum(x1, x2), x1)
x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
test_op_grad((s)->sum(s, x1, x2), ()->sum(x1, x2), x2)

# MxN + Mx1
x1, x2 = Variable(randn(5, 3)), Variable(randn(5))
y = sum(x1, x2)
@test size(y) == (5, 3)
@test all(y.data .== x1.data .+ x2.data)

x1, x2 = Variable(randn(5, 3)), Variable(randn(5))
test_op_grad((s)->sum(s, x1, x2), ()->sum(x1, x2), x1)
x1, x2 = Variable(randn(5, 3)), Variable(randn(5))
test_op_grad((s)->sum(s, x1, x2), ()->sum(x1, x2), x2)

# MxN + Mx1 + MxN
xs = vcat(Variable(randn(5, 3)), Variable(randn(5, 1)), Variable(randn(5, 3)))
y = sum(xs)
y_expected = xs[1].data .+ xs[2].data .+ xs[3].data
@test size(y) == (5, 3)
@test all(y.data .== y_expected)

for k = 1:3
    xs = vcat(Variable(randn(5, 3)), Variable(randn(5, 1)), Variable(randn(5, 3)))
    test_op_grad((s)->sum(s, xs), ()->sum(xs), xs[k])
end

# sum(x1, x2, x3...)
xs = vcat(Variable(randn(5, 3)), Variable(randn(5, 1)), Variable(randn(5, 3)), Variable(randn(5, 1)))
y = sum(xs...)
y_expected = xs[1].data .+ xs[2].data .+ xs[3].data .+ xs[4].data
@test size(y) == (5, 3)
@test all(y.data .== y_expected)

for k = 1:4
    xs = vcat(Variable(randn(5, 3)), Variable(randn(5, 1)), Variable(randn(5, 3)), Variable(randn(5, 1)))
    test_op_grad((s)->sum(s, xs...), ()->sum(xs...), xs[k])
end

# Vector of MxN
xs = [Variable(randn(5, 3)) for i = 1:10]
y = sum(xs)
y_expected = sum([x.data for x in xs])
@test size(y) == (5, 3)
@test all(y.data .== y_expected)

for k = 1:10
    xs = [Variable(randn(5, 3)) for i = 1:10]
    test_op_grad((s)->sum(s, xs), ()->sum(xs), xs[k])
end
