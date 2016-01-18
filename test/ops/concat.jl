using Flimsy
using Base.Test

K = 3
mlb, mub = 2, 7
m = [rand(mlb:mub) for k = 1:K]
n = 1
xs = [GradVariable(randn(m[k], n)) for k = 1:K]
y = concat(xs)
@test size(y) == (sum(m), n)
@test all(vcat([x.data for x in xs]...) .== y.data)

test_op_grad_mse(concat, xs, wrt=xs)


K = 3
mlb, mub = 2, 7
m = [rand(mlb:mub) for k = 1:K]
n = 5
xs = [GradVariable(randn(m[k], n)) for k = 1:K]
y = concat(xs)
@test size(y) == (sum(m), n)
@test all(vcat([x.data for x in xs]...) .== y.data)

test_op_grad_mse(concat, xs, wrt=xs)
