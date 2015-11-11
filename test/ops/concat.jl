using Flimsy
import Flimsy: Var
using Base.Test

xs = [Var(randn(rand(1:10), 3)) for i = 1:10]
y = concat(xs)
@test size(y) == (sum([size(x, 1) for x in xs]), 3)
@test all(y.data .== vcat([x.data for x in xs]...))

for k = 1:10
    xs = [Var(randn(rand(1:10), 3)) for i = 1:10]
    test_op_grad((s)->concat(s, xs), ()->concat(xs), xs[k])
end
