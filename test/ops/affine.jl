using Flimsy

facts("affine") do 
    for (m, n, k) in [(4, 3, 1), (5, 10, 7)]
        context("$(m)x$(n)x$(k)") do
            for W in [Constant, Variable]
                for X in [Constant, Variable]
                    for B in [Constant, Variable]
                        ctxstr = string("{",
                            W <: Constant ? "Constant" : "Variable",
                            ",", 
                            X <: Constant ? "Constant" : "Variable",
                            ",",
                            B <: Constant ? "Constant" : "Variable",
                        "}")
                        context(ctxstr) do
                            ysz = m, k
                            w = W(randn(m, n))
                            x = X(randn(n, k))
                            b = B(randn(m, 1))
                            y = affine(RunScope(), w, x, b)
                            @fact isa(y, Constant) --> true
                            @fact size(y) --> (m, k)
                            @fact y.data --> roughly(w.data * x.data .+ b.data)

                            w = W(randn(m, n))
                            x = X(randn(n, k))
                            b = B(randn(m, 1))
                            y = affine(GradScope(), w, x, b)
                            if any(T -> T <: Variable, [W, X, B])
                                @fact isa(y, Variable) --> true
                            else
                                @fact isa(y, Constant) --> true
                            end
                            @fact size(y) --> (m, k)
                            @fact y.data  --> roughly(w.data * x.data .+ b.data)

                            if any(T -> T <: Variable, [W, X, B])
                                w = W(randn(m, n))
                                x = X(randn(n, k))
                                b = B(randn(m, 1))
                                wrt = []
                                isa(w, Variable) && push!(wrt, w)
                                isa(x, Variable) && push!(wrt, x)
                                isa(b, Variable) && push!(wrt, b) 
                                test_op_grad_mse(affine, w, x, b, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
