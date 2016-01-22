using Flimsy
using Base.Test

for a in [1.5, 0.5, -0.5, 1.5]
    # minus(Real, Mx1)
    m, n = 3, 1
    x = DataVariable(randn(m))
    y = minus(a, x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq a .- x.data y.data

    x = GradVariable(randn(m))
    y = minus(a, x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq a .- x.data y.data

    x = DataVariable(randn(m))
    y = minus(CallbackStack(), a, x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq a .- x.data y.data

    x = GradVariable(randn(m))
    y = minus(CallbackStack(), a, x)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test_approx_eq a .- x.data y.data

    x = GradVariable(randn(m))
    test_op_grad_mse(minus, a, x, wrt=x)

    # minus(MxN, Real)
    x = DataVariable(randn(m))
    y = minus(x, a)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq x.data .- a y.data

    x = GradVariable(randn(m))
    y = minus(x, a)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq x.data .- a y.data

    x = DataVariable(randn(m))
    y = minus(CallbackStack(), x, a)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq x.data .- a y.data

    x = GradVariable(randn(m))
    y = minus(CallbackStack(), x, a)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test_approx_eq x.data .- a y.data

    x = GradVariable(randn(m))
    test_op_grad_mse(minus, x, a, wrt=x)

    # minus(Real, MxN)
    m, n = 5, 8
    x = DataVariable(randn(m, n))
    y = minus(a, x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq a .- x.data y.data

    x = GradVariable(randn(m, n))
    y = minus(a, x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq a .- x.data y.data

    x = DataVariable(randn(m, n))
    y = minus(CallbackStack(), a, x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq a .- x.data y.data

    x = GradVariable(randn(m, n))
    y = minus(CallbackStack(), a, x)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test_approx_eq a .- x.data y.data

    x = GradVariable(randn(m, n))
    test_op_grad_mse(minus, a, x, wrt=x)

    # minus(MxN, Real)
    x = DataVariable(randn(m, n))
    y = minus(x, a)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq x.data .- a y.data

    x = GradVariable(randn(m, n))
    y = minus(x, a)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq x.data .- a y.data

    x = DataVariable(randn(m, n))
    y = minus(CallbackStack(), x, a)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test_approx_eq x.data .- a y.data

    x = GradVariable(randn(m, n))
    y = minus(CallbackStack(), x, a)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test_approx_eq x.data .- a y.data

    x = GradVariable(randn(m, n))
    test_op_grad_mse(minus, x, a, wrt=x)

end
