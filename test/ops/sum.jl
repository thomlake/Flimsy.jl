using Flimsy
using Base.Test

# Binary Sums
# Mx1 + Mx1
m, n = 3, 1
a = GradVariable(randn(m))
b = GradVariable(randn(m))
c = sum(a, b)
@test size(c) == (m, n)
@test all((a.data .+ b.data) .== c.data)
test_op_grad_mse(sum, a, b, wrt=[a, b])


# MxN + MxN
m, n = 3, 5
a = GradVariable(randn(m, n))
b = GradVariable(randn(m, n))
c = sum(a, b)
@test size(c) == (m, n)
@test all((a.data .+ b.data) .== c.data)
test_op_grad_mse(sum, a, b, wrt=[a, b])


# MxN + Mx1
m, n = 3, 5
a = GradVariable(randn(m, n))
b = GradVariable(randn(m))
c = sum(a, b)
@test size(c) == (m, n)
@test all((a.data .+ b.data) .== c.data)
test_op_grad_mse(sum, a, b, wrt=[a, b])

# Mx1 + MxN
m, n = 3, 5
a = GradVariable(randn(m))
b = GradVariable(randn(m, n))
c = sum(a, b)
@test size(c) == (m, n)
@test all((a.data .+ b.data) .== c.data)
test_op_grad_mse(sum, a, b, wrt=[a, b])


# K-ary Sum
# Mx1 + ... + Mx1
K, m, n = 5, 3, 1
xs = [GradVariable(randn(m)) for k = 1:K]
y = sum(xs)
@test size(y) == (m, n)
@test all(sum(map(x->x.data, xs)) .== y.data)
test_op_grad_mse(sum, xs, wrt=xs)


# MxN + ... + MxN
K, m, n = 10, 3, 5
xs = [GradVariable(randn(m, n)) for k = 1:K]
y = sum(xs)
@test size(y) == (m, n)
@test all(sum(map(x->x.data, xs)) .== y.data)
test_op_grad_mse(sum, xs, wrt=xs)


# MxN + Mx1 + ... + Mx1
K, m, n = 6, 3, 5
xs = [GradVariable(randn(m, n))]
for k = 2:K
    push!(xs, GradVariable(randn(m)))
end
y = sum(xs)
@test size(y) == (m, n)
y_true = deepcopy(xs[1].data)
for k = 2:K
    y_true = y_true .+ xs[k].data
end 
@test all(y_true .== y.data)
test_op_grad_mse(sum, xs, wrt=xs)
