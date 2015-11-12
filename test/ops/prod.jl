using Flimsy
using Base.Test

# Constant * Mx1
c = Float64(pi)
x = Variable(randn(3))
y = prod(c, x)
@test size(y) == (3, 1)
for i in eachindex(y.data)
    @test y.data[i] == c * x.data[i]
end
test_op_grad((s)->prod(s, c, x), ()->prod(c, x), x)

c = -Float64(pi)
x = Variable(randn(3))
y = prod(c, x)
@test size(y) == (3, 1)
for i in eachindex(y.data)
    @test y.data[i] == c * x.data[i]
end
test_op_grad((s)->prod(s, c, x), ()->prod(c, x), x)


# Constant * MxN
c = Float64(pi)
x = Variable(randn(3, 5))
y = prod(c, x)
@test size(y) == (3, 5)
for i in eachindex(y.data)
    @test y.data[i] == c * x.data[i]
end
test_op_grad((s)->prod(s, c, x), ()->prod(c, x), x)

c = -Float64(pi)
x = Variable(randn(3, 5))
y = prod(c, x)
@test size(y) == (3, 5)
for i in eachindex(y.data)
    @test y.data[i] == c * x.data[i]
end
test_op_grad((s)->prod(s, c, x), ()->prod(c, x), x)


# Mx1 * Mx1
x1, x2 = Variable(randn(5)), Variable(randn(5))
y = prod(x1, x2)
@test size(y) == (5, 1)
for i in eachindex(y.data)
    @test y.data[i] == x1.data[i] * x2.data[i]
end
x1, x2 = Variable(randn(5)), Variable(randn(5))
test_op_grad((s)->prod(s, x1, x2), ()->prod(x1, x2), x1)
x1, x2 = Variable(randn(5)), Variable(randn(5))
test_op_grad((s)->prod(s, x1, x2), ()->prod(x1, x2), x2)


# MxN * MxN
x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
y = prod(x1, x2)
@test size(y) == (5, 3)
for j = 1:size(y, 2)
    for i = 1:size(y, 1)
        @test y.data[i,j] == x1.data[i,j] * x2.data[i,j]
    end
end
x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
test_op_grad((s)->prod(s, x1, x2), ()->prod(x1, x2), x1)
x1, x2 = Variable(randn(5, 3)), Variable(randn(5, 3))
test_op_grad((s)->prod(s, x1, x2), ()->prod(x1, x2), x2)


# 1xN * MxN
x1, x2 = Variable(randn(1, 3)), Variable(randn(5, 3))
y = prod(x2, x1)
@test size(y) == (5, 3)
for j = 1:size(y, 2)
    for i = 1:size(y, 1)
        @test y.data[i,j] == x1.data[j] * x2.data[i,j]
    end
end
x1, x2 = Variable(randn(1, 3)), Variable(randn(5, 3))
test_op_grad((s)->prod(s, x1, x2), ()->prod(x1, x2), x1)
x1, x2 = Variable(randn(1, 3)), Variable(randn(5, 3))
test_op_grad((s)->prod(s, x1, x2), ()->prod(x1, x2), x2)
