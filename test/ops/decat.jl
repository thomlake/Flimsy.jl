using Flimsy
using Base.Test

f(x) = concat(decat(x))
f(s, x) = concat(s, decat(s, x))

m, n = 6, 1
x = GradVariable(randn(m))
ys = decat(x)
for i = 1:m
    @test size(ys[i]) == (1, n)
    for j = 1:n
        @test all(x.data[i,j] .== ys[i].data[j])
    end
end
test_op_grad_mse(f, x, wrt=x)

m, n = 3, 9
x = GradVariable(randn(m, n))
ys = decat(x)
for i = 1:m
    @test size(ys[i]) == (1, n)
    for j = 1:n
        @test all(x.data[i,j] .== ys[i].data[j])
    end
end
test_op_grad_mse(f, x, wrt=x)
