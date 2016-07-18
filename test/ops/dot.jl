using Flimsy

facts("dot") do
    for (m, n) in [(1, 1), (1, 4), (4, 1), (4, 3), (4, 5)]
        context(string(m, "x", n)) do
            for A in [Constant, Variable]
                for B in [Constant, Variable]
                    ctxstr = string("{",
                        A <: Constant ? "Constant" : "Variable",
                        ",",
                        B <: Constant ? "Constant" : "Variable",
                    "}")
                    context(ctxstr) do
                        a = A(randn(m, n))
                        b = B(randn(m, n))

                        c = dot(RunScope(), a, b)
                        @fact isa(c, Constant) --> true
                        @fact size(c) --> (1, n)
                        for j = 1:n
                            @fact c.data[1,j] --> dot(a.data[:,j], b.data[:,j])
                        end

                        c = dot(GradScope(), a, b)
                        if any(T -> T <: Variable, [A, B])
                            @fact isa(c, Variable) --> true
                        else
                            @fact isa(c, Constant) --> true
                        end
                        @fact size(c) --> (1, n)
                        for j = 1:n
                            @fact c.data[1,j] --> dot(a.data[:,j], b.data[:,j])
                        end

                        if any(T -> T <: Variable, [A, B])
                            context("Gradient") do
                                wrt = []
                                a = A(randn(m, n))
                                b = B(randn(m, n))
                                isa(a, Variable) && push!(wrt, a)
                                isa(b, Variable) && push!(wrt, b)
                                test_op_grad_mse(dot, a, b, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
