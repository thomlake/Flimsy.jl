using Flimsy
using Base.Test

# m 1x1 matrices
m, n = 8, 1
xmat = randn(m, n)
ymat = softmax(xmat)
xs = [GradVariable(xmat[i,:]) for i = 1:m]
ys = softmax(xs)
@test length(ys) == m
for i = 1:m
    @test size(ys[i]) == (1, n)
    @test all(ymat[i,:] .== ys[i].data)
end

for j = 1:n
    s = 0.0
    for i = 1:m
        s += ys[i].data[j]
    end
    @test isapprox(s, 1)
end

f(s, xs) = concat(s, softmax(s, xs))
f(xs) = concat(softmax(xs))
test_op_grad_mse(f, xs, wrt=xs)

# m 1x8 matrices
m, n = 5, 8
xmat = randn(m, n)
ymat = softmax(xmat)
xs = [GradVariable(xmat[i,:]) for i = 1:m]
ys = softmax(xs)
@test length(ys) == m
for i = 1:m
    @test size(ys[i]) == (1, n)
    @test all(ymat[i,:] .== ys[i].data)
end

for j = 1:n
    s = 0.0
    for i = 1:m
        s += ys[i].data[j]
    end
    @test isapprox(s, 1)
end

f(s, xs) = concat(s, softmax(s, xs))
f(xs) = concat(softmax(xs))
test_op_grad_mse(f, xs, wrt=xs)
