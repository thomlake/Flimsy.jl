using Flimsy
using Base.Test

x = Variable(randn(3))
y = softmax(x)
@test size(y) == (3, 1)
s = exp(x.data - maximum(x.data))
s = s ./ sum(s)
@test all(s  .== y.data)
@test_approx_eq(sum(y.data), 1)
test_op_grad((s)->softmax(s, x), ()->softmax(x), x)

x = Variable(randn(3, 5))
y = softmax(x)
@test size(y) == (3, 5)
for j = 1:5
    s = exp(x.data[:,j] - maximum(x.data[:,j]))
    s = s ./ sum(s)
    @test all(s  .== y.data[:,j])
    @test_approx_eq(sum(y.data[:,j]), 1)
end
test_op_grad((s)->softmax(s, x), ()->softmax(x), x)

x = [Variable(randn()) for i = 1:10]
y = softmax(x)
@test length(y) == 10
sum_y = 0.0
x_max = maximum([x[i].data[1] for i = 1:10])
sum_exp_x = 0.0
for i = 1:10
    sum_exp_x += exp(x[i].data[1] - x_max)
    sum_y += y[i].data[1]
end

for i = 1:10
    s_i = exp(x[i].data[1] - x_max) / sum_exp_x
    @test all(s_i == y[i].data[1])
end
@test_approx_eq(sum_y, 1)

test_op_grad((s)->softmax(s, x), ()->softmax(x), x)
