using Flimsy
using Iterators

facts("plus_cols") do
    for (m, n) in [(3, 1), (4, 5)]
        context("$(m)x$(n)") do
            for A in [Constant, Variable]
                for B in [Constant, Variable]
                    ctxstring = string("{", 
                        (A <: Constant ? "Constant" : "Variable"), 
                        ",", 
                        (B <: Constant ? "Constant" : "Variable"), 
                    "}")
                    context(ctxstring) do
                        a = A(randn(m, n))
                        b = B(randn(m, 1))
                        c = plus_cols(RunScope(), a, b)
                        @fact isa(c, Constant) --> true
                        @fact size(c)          --> (m, n)
                        @fact c.data           --> roughly(a.data .+ b.data)

                        c = plus_cols(GradScope(), a, b)
                        if any(T -> T <: Variable, [A, B])
                            @fact isa(c, Variable) --> true
                        else
                            @fact isa(c, Constant) --> true
                        end
                        @fact size(c) --> (m, n)
                        @fact c.data  --> roughly(a.data .+ b.data)

                        if any(T -> T <: Variable, [A, B])
                            context("Gradient") do
                                a = A(randn(m, n))
                                b = B(randn(m, 1))
                                wrt = []
                                isa(a, Variable) && push!(wrt, a)
                                isa(b, Variable) && push!(wrt, b)
                                test_op_grad_mse(plus_cols, a, b, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
