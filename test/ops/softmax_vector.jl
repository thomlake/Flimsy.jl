using Flimsy
using Base.Test

# M 1x1 matrices
function test_1x1()
    m, n = 8, 1
    xmat = randn(m, n)
    ymat = softmax(xmat)
    xs = [DataVariable(xmat[i,:]) for i = 1:m]
    ys = softmax(xs)
    @test length(ys) == m
    @test isa(ys, Vector)
    @test eltype(ys) <: DataVariable
    for i = 1:m
        @test size(ys[i]) == (1, n)
        @test_approx_eq ymat[i,:] ys[i].data
    end

    for j = 1:n
        s = 0.0
        for i = 1:m
            s += ys[i].data[j]
        end
        @test isapprox(s, 1)
    end

    xmat = randn(m, n)
    ymat = softmax(xmat)
    xs = [GradVariable(xmat[i,:]) for i = 1:m]
    ys = softmax(xs)
    @test length(ys) == m
    @test isa(ys, Vector)
    @test eltype(ys) <: DataVariable
    for i = 1:m
        @test size(ys[i]) == (1, n)
        @test_approx_eq ymat[i,:] ys[i].data
    end

    for j = 1:n
        s = 0.0
        for i = 1:m
            s += ys[i].data[j]
        end
        @test isapprox(s, 1)
    end

    xmat = randn(m, n)
    ymat = softmax(xmat)
    xs = [DataVariable(xmat[i,:]) for i = 1:m]
    ys = softmax(CallbackStack(), xs)
    @test length(ys) == m
    @test isa(ys, Vector)
    @test eltype(ys) <: DataVariable
    for i = 1:m
        @test size(ys[i]) == (1, n)
        @test_approx_eq ymat[i,:] ys[i].data
    end

    for j = 1:n
        s = 0.0
        for i = 1:m
            s += ys[i].data[j]
        end
        @test isapprox(s, 1)
    end

    xmat = randn(m, n)
    ymat = softmax(xmat)
    xs = [GradVariable(xmat[i,:]) for i = 1:m]
    ys = softmax(CallbackStack(), xs)
    @test length(ys) == m
    @test isa(ys, Vector)
    @test eltype(ys) <: GradVariable
    for i = 1:m
        @test size(ys[i]) == (1, n)
        @test_approx_eq ymat[i,:] ys[i].data
    end

    for j = 1:n
        s = 0.0
        for i = 1:m
            s += ys[i].data[j]
        end
        @test isapprox(s, 1)
    end

    xmat = randn(m, n)
    xs = [GradVariable(xmat[i,:]) for i = 1:m]
    f(s, xs) = concat(s, softmax(s, xs))
    f(xs) = concat(softmax(xs))
    test_op_grad_mse(f, xs, wrt=xs)
end
test_1x1()

# M 1xN matrices
function test_1xN()
    m, n = 5, 8
    xmat = randn(m, n)
    ymat = softmax(xmat)
    xs = [DataVariable(xmat[i,:]) for i = 1:m]
    ys = softmax(xs)
    @test length(ys) == m
    @test isa(ys, Vector)
    @test eltype(ys) <: DataVariable
    for i = 1:m
        @test size(ys[i]) == (1, n)
        @test_approx_eq ymat[i,:] ys[i].data
    end

    for j = 1:n
        s = 0.0
        for i = 1:m
            s += ys[i].data[j]
        end
        @test isapprox(s, 1)
    end

    xmat = randn(m, n)
    ymat = softmax(xmat)
    xs = [GradVariable(xmat[i,:]) for i = 1:m]
    ys = softmax(xs)
    @test length(ys) == m
    @test isa(ys, Vector)
    @test eltype(ys) <: DataVariable
    for i = 1:m
        @test size(ys[i]) == (1, n)
        @test_approx_eq ymat[i,:] ys[i].data
    end

    for j = 1:n
        s = 0.0
        for i = 1:m
            s += ys[i].data[j]
        end
        @test isapprox(s, 1)
    end

    xmat = randn(m, n)
    ymat = softmax(xmat)
    xs = [DataVariable(xmat[i,:]) for i = 1:m]
    ys = softmax(CallbackStack(), xs)
    @test length(ys) == m
    @test isa(ys, Vector)
    @test eltype(ys) <: DataVariable
    for i = 1:m
        @test size(ys[i]) == (1, n)
        @test_approx_eq ymat[i,:] ys[i].data
    end

    for j = 1:n
        s = 0.0
        for i = 1:m
            s += ys[i].data[j]
        end
        @test isapprox(s, 1)
    end

    xmat = randn(m, n)
    ymat = softmax(xmat)
    xs = [GradVariable(xmat[i,:]) for i = 1:m]
    ys = softmax(CallbackStack(), xs)
    @test length(ys) == m
    @test isa(ys, Vector)
    @test eltype(ys) <: GradVariable
    for i = 1:m
        @test size(ys[i]) == (1, n)
        @test_approx_eq ymat[i,:] ys[i].data
    end

    for j = 1:n
        s = 0.0
        for i = 1:m
            s += ys[i].data[j]
        end
        @test isapprox(s, 1)
    end

    xmat = randn(m, n)
    xs = [GradVariable(xmat[i,:]) for i = 1:m]
    f(s, xs) = concat(s, softmax(s, xs))
    f(xs) = concat(softmax(xs))
    test_op_grad_mse(f, xs, wrt=xs)
end
test_1xN()
