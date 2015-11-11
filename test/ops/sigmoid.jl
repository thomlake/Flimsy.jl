using Flimsy
import Flimsy: Var
using Base.Test

x = Var(randn(3))
y = sigmoid(x)
@test size(y) == (3, 1)
@test all(1 ./ (1 + exp(-x.data)) .== y.data)
test_op_grad((s)->sigmoid(s, x), ()->sigmoid(x), x)

x = Var(randn(3, 5))
y = sigmoid(x)
@test size(y) == (3, 5)
@test all(1 ./ (1 + exp(-x.data)) .== y.data)
test_op_grad((s)->sigmoid(s, x), ()->sigmoid(x), x)
