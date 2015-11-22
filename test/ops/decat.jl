using Flimsy
using Base.Test

x = Variable(randn(10))
ys = decat(x)
@test length(ys) == 10
@test reduce(&, [size(y) == (1, 1) for y in ys])
@test reduce(&, [ys[i].data[1] == x.data[i] for i = 1:10])

test_op_grad((s)->concat(s, decat(s, x)), ()->concat(decat(x)), x)
