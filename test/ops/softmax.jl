using Flimsy
import Flimsy: Var
using Base.Test

x = Var(randn(3))
y = softmax(x)
@test size(y) == (3, 1)
s = exp(x.data - maximum(x.data))
s = s ./ sum(s)
@test all(s  .== y.data)
@test_approx_eq(sum(y.data), 1)
test_op_grad((s)->softmax(s, x), ()->softmax(x), x)

x = Var(randn(3, 5))
y = softmax(x)
@test size(y) == (3, 5)
for j = 1:5
    s = exp(x.data[:,j] - maximum(x.data[:,j]))
    s = s ./ sum(s)
    @test all(s  .== y.data[:,j])
    @test_approx_eq(sum(y.data[:,j]), 1)
end
test_op_grad((s)->softmax(s, x), ()->softmax(x), x)
