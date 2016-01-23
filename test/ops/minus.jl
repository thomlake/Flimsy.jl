using Flimsy
using Base.Test

function test_minus()
    for (m, n) in [(3, 1), (4, 5)]
        for atype in [DataVariable, GradVariable]
            for btype in [DataVariable, GradVariable]
                # (MxN, MxN)
                a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
                c = minus(a, b)
                @test isa(c, DataVariable)
                @test size(c) == (m, n)
                @test_approx_eq a.data .- b.data c.data

                a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
                c = minus(CallbackStack(), a, b)
                @test isa(c, anygrads(atype, btype) ? GradVariable : DataVariable)
                @test size(c) == (m, n)
                @test_approx_eq a.data .- b.data c.data

                if anygrads(atype, btype)
                    a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
                    wrt = []
                    atype <: GradVariable && push!(wrt, a)
                    btype <: GradVariable && push!(wrt, b)
                    test_op_grad_mse(minus, a, b, wrt=wrt)
                end

                # minus(1xN, MxN)
                a = atype(randn(1, n))
                b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
                c = minus(a, b)
                @test isa(c, DataVariable)
                @test size(c) == (m, n)
                @test_approx_eq a.data .- b.data c.data

                a = atype(randn(1, n))
                b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
                c = minus(CallbackStack(), a, b)
                @test isa(c, anygrads(atype, btype) ? GradVariable : DataVariable)
                @test size(c) == (m, n)
                @test_approx_eq a.data .- b.data c.data

                if anygrads(atype, btype)
                    a = atype(randn(1, n))
                    b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
                    wrt = []
                    atype <: GradVariable && push!(wrt, a)
                    btype <: GradVariable && push!(wrt, b)
                    test_op_grad_mse(minus, a, b, wrt=wrt)
                end

                # minus(Mx1, MxN)
                a = atype(randn(m, 1))
                b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
                c = minus(a, b)
                @test isa(c, DataVariable)
                @test size(c) == (m, n)
                @test_approx_eq a.data .- b.data c.data

                a = atype(randn(m, 1))
                b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
                c = minus(CallbackStack(), a, b)
                @test isa(c, anygrads(atype, btype) ? GradVariable : DataVariable)
                @test size(c) == (m, n)
                @test_approx_eq a.data .- b.data c.data

                if anygrads(atype, btype)
                    a = atype(randn(m, 1))
                    b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
                    wrt = []
                    atype <: GradVariable && push!(wrt, a)
                    btype <: GradVariable && push!(wrt, b)
                    test_op_grad_mse(minus, a, b, wrt=wrt)
                end
            end
        end
    end
end
test_minus()
