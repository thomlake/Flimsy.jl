using Flimsy

facts("linear") do
    for (wsz, xsz) in [((1, 4), (4, 1)), ((4, 3), (3, 1)), ((1, 4), (4, 5)), ((3, 6), (6, 7))]
        context(string(join(wsz, "x"), " * ", join(xsz, "x"))) do
            for W in [Constant, Variable]
                for X in [Constant, Variable]
                    ctxstr = string(
                        W <: Constant ? "Constant" : "Variable",
                        ",", 
                        X <: Constant ? "Constant" : "Variable",
                    )
                    context(ctxstr) do
                        ysz = (wsz[1], xsz[2])
                        w = W(randn(wsz))
                        x = X(randn(xsz))
                        y = linear(RunScope(), w, x)
                        @fact isa(y, Constant) --> true
                        @fact size(y) --> ysz
                        @fact y.data --> w.data * x.data

                        w = W(randn(wsz))
                        x = X(randn(xsz))
                        y = linear(GradScope(), w, x)
                        if any(T -> T <: Variable, [W, X])
                            @fact isa(y, Variable) --> true
                        else
                            @fact isa(y, Constant) --> true
                        end
                        @fact size(y) --> ysz
                        @fact y.data --> w.data * x.data

                        if any(T -> T <: Variable, [W, X])
                            context("Gradient") do
                                w = W(randn(wsz))
                                x = X(randn(xsz))
                                wrt = []
                                isa(w, Variable) && push!(wrt, w)
                                isa(x, Variable) && push!(wrt, x)
                                test_op_grad_mse(linear, w, x, wrt=wrt)
                            end
                        end
                    end
                end
            end
        end
    end
end
