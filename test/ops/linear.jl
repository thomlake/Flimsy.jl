using Flimsy
import Flimsy: Var
using Base.Test

# MxN dot Nx1
w, x = Var(randn(5, 3)), Var(randn(3))
y = linear(w, x)
@test size(y) == (5, 1)
@test all(y.data .== w.data * x.data)

w, x = Var(randn(5, 3)), Var(randn(3))
test_op_grad((s)->linear(s, w, x), ()->linear(w, x), w)
w, x = Var(randn(5, 3)), Var(randn(3))
test_op_grad((s)->linear(s, w, x), ()->linear(w, x), x)

# MxK dot KxN
w, x = Var(randn(5, 3)), Var(randn(3, 10))
y = linear(w, x)
@test size(y) == (5, 10)
@test all(y.data .== w.data * x.data)

w, x = Var(randn(5, 3)), Var(randn(3, 10))
test_op_grad((s)->linear(s, w, x), ()->linear(w, x), w)
w, x = Var(randn(5, 3)), Var(randn(3, 10))
test_op_grad((s)->linear(s, w, x), ()->linear(w, x), x)
