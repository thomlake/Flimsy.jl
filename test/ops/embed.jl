using Flimsy
using Distributions

facts("embed") do
    for (wsz, xsz) in [((1, 4), (4, 1)), ((4, 3), (3, 1)), ((1, 4), (4, 5)), ((3, 6), (6, 7))]
        context(string(join(wsz, "x"), " * ", join(xsz, "x"))) do
            for W in [Constant, Variable]
                context(W <: Constant ? "Constant" : "Variable",) do
                    ysz = (wsz[1], xsz[2])
                    w = W(randn(wsz))
                    x = if xsz[2] > 1
                        Vector{Int}[sample(1:xsz[1], rand(1:xsz[1]), replace=false) for i = 1:xsz[2]]
                    else
                        sample(1:xsz[1], rand(1:xsz[1]), replace=false)
                    end
                    x_arr = zeros(FloatX, xsz)
                    if xsz[2] > 1
                        for k = 1:xsz[2]
                            for i in x[k]
                                x_arr[i,k] = 1
                            end
                        end
                    else
                        for i in x
                            x_arr[i] = 1
                        end
                    end
                    y = linear(RunScope(), w, x)
                    @fact isa(y, Constant) --> true
                    @fact size(y) --> ysz
                    @fact y.data --> roughly(w.data * x_arr)

                    
                    w = W(randn(wsz))
                    x = if xsz[2] > 1
                        Vector{Int}[sample(1:xsz[1], rand(1:xsz[1]), replace=false) for i = 1:xsz[2]]
                    else
                        sample(1:xsz[1], rand(1:xsz[1]), replace=false)
                    end
                    x_arr = zeros(FloatX, xsz)
                    if xsz[2] > 1
                        for k = 1:xsz[2]
                            for i in x[k]
                                x_arr[i,k] = 1
                            end
                        end
                    else
                        for i in x
                            x_arr[i] = 1
                        end
                    end
                    y = linear(GradScope(), w, x)
                    @fact isa(y, W) --> true
                    @fact size(y) --> ysz
                    @fact y.data --> roughly(w.data * x_arr)

                    if isa(w, Variable)
                        context("Gradient") do
                            w = W(randn(wsz))
                            x = if xsz[2] > 1
                                Vector{Int}[sample(1:xsz[1], rand(1:xsz[1]), replace=false) for i = 1:xsz[2]]
                            else
                                sample(1:xsz[1], rand(1:xsz[1]), replace=false)
                            end
                            test_op_grad_mse(linear, w, x, wrt=w)
                        end
                    end
                end
            end
        end
    end
end
