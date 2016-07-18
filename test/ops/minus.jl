using Flimsy

facts("minus (primative)") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            for a in [1.5, 0.5, -0.5, 1.5]
                for T in [Constant, Variable]
                    b = T(randn(m, n))
                    c = minus(RunScope(), a, b)
                    @fact isa(c, Constant) --> true
                    @fact size(c) --> (m, n)
                    for i in eachindex(c)
                        @fact c.data[i] --> roughly(FloatX(a - b.data[i]))
                    end

                    b = T(randn(m, n))
                    c = minus(GradScope(), a, b)
                    if T <: Variable
                        @fact isa(c, Variable) --> true
                    else
                        @fact isa(c, Constant) --> true
                    end
                    @fact size(c) --> (m, n)
                    for i in eachindex(c)
                        @fact c.data[i] --> roughly(FloatX(a - b.data[i]))
                    end
                end

                b = Variable(randn(m, n))
                test_op_grad_mse(minus, a, b, wrt=b)
            end
        end
    end
end

facts("minus") do
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
                        c = minus(RunScope(), a, b)
                        @fact isa(c, Constant) --> true
                        @fact size(c) --> (m, n)
                        for i in eachindex(c)
                            @fact c.data[i] --> roughly(FloatX(a.data[i] - b.data[i]))
                        end

                        c = minus(GradScope(), a, b)
                        if any(T -> T <: Variable, [A, B])
                            @fact isa(c, Variable) --> true
                        else
                            @fact isa(c, Constant) --> true
                        end
                        @fact size(c) --> (m, n)
                        for i in eachindex(c)
                            @fact c.data[i] --> roughly(FloatX(a.data[i] - b.data[i]))
                        end

                        if any(T -> T <: Variable, [A, B])
                            context("Gradient") do
                                a = A(randn(m, n))
                                b = B(randn(m, n))
                                wrt = []
                                isa(a, Variable) && push!(wrt, a)
                                isa(b, Variable) && push!(wrt, b)
                                test_op_grad_mse(minus, a, b, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
