using Flimsy
using Iterators

facts("plus") do
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
                        b = B(randn(m, n))
                        c = plus(RunScope(), a, b)
                        @fact isa(c, Constant) --> true
                        @fact size(c) --> (m, n)
                        for i in eachindex(c)
                            @fact c.data[i] --> roughly(a.data[i] + b.data[i])
                        end

                        c = plus(GradScope(), a, b)
                        if any(T -> T <: Variable, [A, B])
                            @fact isa(c, Variable) --> true
                        else
                            @fact isa(c, Constant) --> true
                        end
                        @fact size(c) --> (m, n)
                        for i in eachindex(c)
                            @fact c.data[i] --> roughly(a.data[i] + b.data[i])
                        end

                        if any(T -> T <: Variable, [A, B])
                            context("Gradient") do
                                a = A(randn(m, n))
                                b = B(randn(m, n))
                                wrt = []
                                isa(a, Variable) && push!(wrt, a)
                                isa(b, Variable) && push!(wrt, b)
                                test_op_grad_mse(plus, a, b, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
