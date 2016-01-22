using Flimsy
using Base.Test

function test_wta()
    # Mx1
    m, n = 3, 1
    x = DataVariable(randn(m))
    y = wta(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(wta(x.data) .== y.data)
    @test countnz(y.data) == n

    x = GradVariable(randn(m))
    y = wta(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(wta(x.data) .== y.data)
    @test countnz(y.data) == n

    x = DataVariable(randn(m))
    y = wta(CallbackStack(), x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(wta(x.data) .== y.data)
    @test countnz(y.data) == n

    x = GradVariable(randn(m))
    y = wta(CallbackStack(), x)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test all(wta(x.data) .== y.data)
    @test countnz(y.data) == n

    x = GradVariable(randn(m))
    test_op_grad_mse(wta, x, wrt=x)

    # MxN
    m, n = 5, 8
    x = DataVariable(randn(m, n))
    y = wta(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(wta(x.data) .== y.data)
    @test countnz(y.data) == n

    x = GradVariable(randn(m, n))
    y = wta(x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(wta(x.data) .== y.data)
    @test countnz(y.data) == n

    x = DataVariable(randn(m, n))
    y = wta(CallbackStack(), x)
    @test isa(y, DataVariable)
    @test size(y) == (m, n)
    @test all(wta(x.data) .== y.data)
    @test countnz(y.data) == n

    x = GradVariable(randn(m, n))
    y = wta(CallbackStack(), x)
    @test isa(y, GradVariable)
    @test size(y) == (m, n)
    @test all(wta(x.data) .== y.data)
    @test countnz(y.data) == n

    x = GradVariable(randn(m, n))
    test_op_grad_mse(wta, x, wrt=x)
end
test_wta()
