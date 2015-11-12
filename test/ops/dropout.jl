using Flimsy
using Base.Test

x = Variable(randn(3))
y = dropout!(x, 1.0)
@test size(y) == (3, 1)
@test all(y.data .== 0.0)

x = Variable(randn(3))
y = dropout!(x, 0.0)
@test size(y) == (3, 1)
@test all(y.data .== x.data)

x = Variable(ones(300, 200))
y = dropout!(x, 0.5)
@test size(y) == (300, 200)
@test abs(sum(y.data) - 0.5 * 300 * 200) < 1000

x = Variable(ones(300, 200))
y = dropout!(x, 0.2)
@test size(y) == (300, 200)
@test abs(sum(y.data) - 0.8 * 300 * 200) < 1000

x = Variable(ones(300, 200))
y = dropout!(x, 0.8)
@test size(y) == (300, 200)
@test abs(sum(y.data) - 0.2 * 300 * 200) < 1000
