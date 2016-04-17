using Flimsy

facts("minus (binary)") do
    for (m, n) in [(3, 1), (4, 5)]
        context("$(m)x$(n)") do
            for atype in [DataVariable, GradVariable]
                for btype in [DataVariable, GradVariable]
                    ctxstr = string(
                        atype <: DataVariable ? "DataVariable" : "GradVariable",
                        ",", 
                        btype <: DataVariable ? "DataVariable" : "GradVariable",
                    )
                    context(ctxstr) do
                        context("MxN + MxN") do
                            scope = DataScope()
                            gscope = GradScope()

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = minus(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .- b.data)

                            a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = minus(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .- b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, n)) : atype(randn(m, n), zeros(m, n))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(minus, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("1xN + MxN") do
                            scope = DataScope()
                            gscope = GradScope()

                            a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = minus(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .- b.data)

                            a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = minus(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .- b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(1, n)) : atype(randn(1, n), zeros(1, n))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(minus, a, b, wrt=wrt)
                                end
                            end
                        end

                        context("Mx1 + MxN") do
                            scope = DataScope()
                            gscope = GradScope()

                            a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = minus(scope, a, b)
                            @fact isa(c, DataVariable) --> true
                            @fact size(c)              --> (m, n)
                            @fact c.data               --> roughly(a.data .- b.data)

                            a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                            b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                            c = minus(gscope, a, b)
                            if anygrads(atype, btype)
                                @fact isa(c, GradVariable) --> true
                            else
                                @fact isa(c, DataVariable) --> true
                            end
                            @fact size(c) --> (m, n)
                            @fact c.data  --> roughly(a.data .- b.data)

                            if anygrads(atype, btype)
                                context("Gradient") do
                                    a = atype <: DataVariable ? atype(randn(m, 1)) : atype(randn(m, 1), zeros(m, 1))
                                    b = btype <: DataVariable ? btype(randn(m, n)) : btype(randn(m, n), zeros(m, n))
                                    wrt = []
                                    atype <: GradVariable && push!(wrt, a)
                                    btype <: GradVariable && push!(wrt, b)
                                    test_op_grad_mse(minus, a, b, wrt=wrt)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


# using Base.Test

# function test_minus()
#     for (m, n) in [(3, 1), (4, 5)]
#         for atype in [DataVariable, GradVariable]
#             for btype in [DataVariable, GradVariable]
#                 # (MxN, MxN)
#                 a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
#                 c = minus(a, b)
#                 @test isa(c, DataVariable)
#                 @test size(c) == (m, n)
#                 @test_approx_eq a.data .- b.data c.data

#                 a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
#                 c = minus(CallbackStack(), a, b)
#                 @test isa(c, anygrads(atype, btype) ? GradVariable : DataVariable)
#                 @test size(c) == (m, n)
#                 @test_approx_eq a.data .- b.data c.data

#                 if anygrads(atype, btype)
#                     a, b = n == 1 ? (atype(randn(m)), btype(randn(m))) : (atype(randn(m, n)), btype(randn(m, n)))
#                     wrt = []
#                     atype <: GradVariable && push!(wrt, a)
#                     btype <: GradVariable && push!(wrt, b)
#                     test_op_grad_mse(minus, a, b, wrt=wrt)
#                 end

#                 # minus(1xN, MxN)
#                 a = atype(randn(1, n))
#                 b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
#                 c = minus(a, b)
#                 @test isa(c, DataVariable)
#                 @test size(c) == (m, n)
#                 @test_approx_eq a.data .- b.data c.data

#                 a = atype(randn(1, n))
#                 b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
#                 c = minus(CallbackStack(), a, b)
#                 @test isa(c, anygrads(atype, btype) ? GradVariable : DataVariable)
#                 @test size(c) == (m, n)
#                 @test_approx_eq a.data .- b.data c.data

#                 if anygrads(atype, btype)
#                     a = atype(randn(1, n))
#                     b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
#                     wrt = []
#                     atype <: GradVariable && push!(wrt, a)
#                     btype <: GradVariable && push!(wrt, b)
#                     test_op_grad_mse(minus, a, b, wrt=wrt)
#                 end

#                 # minus(Mx1, MxN)
#                 a = atype(randn(m, 1))
#                 b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
#                 c = minus(a, b)
#                 @test isa(c, DataVariable)
#                 @test size(c) == (m, n)
#                 @test_approx_eq a.data .- b.data c.data

#                 a = atype(randn(m, 1))
#                 b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
#                 c = minus(CallbackStack(), a, b)
#                 @test isa(c, anygrads(atype, btype) ? GradVariable : DataVariable)
#                 @test size(c) == (m, n)
#                 @test_approx_eq a.data .- b.data c.data

#                 if anygrads(atype, btype)
#                     a = atype(randn(m, 1))
#                     b = n == 1 ? btype(randn(m)) : btype(randn(m, n))
#                     wrt = []
#                     atype <: GradVariable && push!(wrt, a)
#                     btype <: GradVariable && push!(wrt, b)
#                     test_op_grad_mse(minus, a, b, wrt=wrt)
#                 end
#             end
#         end
#     end
# end
# test_minus()
