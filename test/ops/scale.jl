using Flimsy

facts("scale (primative)") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            for a in [1.5, 0.5, -0.5, 1.5]
                for T in [Constant, Variable]
                    b = T(randn(m, n))
                    c = scale(RunScope(), a, b)
                    @fact isa(c, Constant) --> true
                    @fact size(c)          --> (m, n)
                    @fact c.data           --> roughly(a .* b.data)

                    b = T(randn(m, n))
                    c = scale(GradScope(), a, b)
                    if T <: Variable
                        @fact isa(c, Variable) --> true
                    else
                        @fact isa(c, Constant) --> true
                    end
                    @fact size(c) --> (m, n)
                    @fact c.data  --> roughly(a .* b.data)
                end

                b = Variable(randn(m, n))
                test_op_grad_mse(scale, a, b, wrt=b)
            end
        end
    end
end

facts("scale") do
    for (m, n) in [(3, 1), (5, 8)]
        context("$(m)x$(n)") do
            for A in [Constant, Variable]
                for B in [Constant, Variable]
                    ctxstring = string("{", 
                        (A <: Constant ? "Constant" : "Variable"), 
                        ",", 
                        (B <: Constant ? "Constant" : "Variable"), 
                    "}")
                    context(ctxstring) do
                        a = A(randn(1, n))
                        b = B(randn(m, n))
                        
                        c = scale(RunScope(), a, b)
                        @fact isa(c, Constant) --> true
                        @fact size(c)          --> (m, n)
                        for j = 1:n
                            @fact c.data[:,j] --> roughly(a.data[j] .* b.data[:,j])
                        end

                        c = scale(GradScope(), a, b)
                        if any(T -> T <: Variable, [A, B])
                            @fact isa(c, Variable) --> true
                        else
                            @fact isa(c, Constant) --> true
                        end
                        @fact size(c) --> (m, n)
                        for j = 1:n
                            @fact c.data[:,j] --> roughly(a.data[j] .* b.data[:,j])
                        end

                        if any(T -> T <: Variable, [A, B])
                            a = A(randn(1, n))
                            b = B(randn(m, n))
                            wrt = Variable[]
                            isa(a, Variable) && push!(wrt, a)
                            isa(b, Variable) && push!(wrt, b)
                            test_op_grad_mse(scale, a, b, wrt=wrt)
                        end
                    end
                end
            end
        end
    end
end
