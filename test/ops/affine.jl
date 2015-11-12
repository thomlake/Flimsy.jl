using Flimsy
using Base.Test

w, x, b = Variable(randn(5, 3)), Variable(randn(3)), Variable(randn(5))
y = affine(w, x, b)
@test size(y) == (5, 1)
@test all(y.data .== (w.data * x.data .+ b.data))


w, x, b = Variable(randn(5, 3)), Variable(randn(3)), Variable(randn(5))
test_op_grad((s)->affine(s, w, x, b), ()->affine(w, x, b), w)

w, x, b = Variable(randn(5, 3)), Variable(randn(3)), Variable(randn(5))
test_op_grad((s)->affine(s, w, x, b), ()->affine(w, x, b), x)

w, x, b = Variable(randn(5, 3)), Variable(randn(3)), Variable(randn(5))
test_op_grad((s)->affine(s, w, x, b), ()->affine(w, x, b), b)


w, x, b = Variable(randn(5, 3)), Variable(randn(3, 10)), Variable(randn(5))
y = affine(w, x, b)
@test size(y) == (5, 10)
@test all(y.data .== (w.data * x.data .+ b.data))

w, x, b = Variable(randn(5, 3)), Variable(randn(3, 10)), Variable(randn(5))
test_op_grad((s)->affine(s, w, x, b), ()->affine(w, x, b), w)

w, x, b = Variable(randn(5, 3)), Variable(randn(3, 10)), Variable(randn(5))
test_op_grad((s)->affine(s, w, x, b), ()->affine(w, x, b), x)

w, x, b = Variable(randn(5, 3)), Variable(randn(3, 10)), Variable(randn(5))
test_op_grad((s)->affine(s, w, x, b), ()->affine(w, x, b), b)
