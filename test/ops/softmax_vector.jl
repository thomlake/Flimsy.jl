using Flimsy

facts("softmax_vector") do
    const stol = 1e-5
    for (m, n) in [(8, 1), (5, 8)]
        context("$(m)x1x$(n)") do
            context("RunScope") do
                scope = RunScope()
                xmat = randn(m, n)
                ymat = softmax(xmat)
                xs = map(i -> Constant(xmat[i,:]), 1:m)
                ys = softmax(scope, xs)
                @fact length(ys)             --> m
                @fact isa(ys, Vector)        --> true
                @fact eltype(ys) <: Constant --> true
                for i = 1:m
                    @fact size(ys[i]) --> (1, n)
                    @fact ys[i].data  --> roughly(map(FloatX, ymat[i,:]))
                end

                for j = 1:n
                    s = 0.0
                    for i = 1:m
                        s += ys[i].data[j]
                    end
                    @fact s --> roughly(1, stol)
                end

                xmat = randn(m, n)
                ymat = softmax(xmat)
                xs = map(i -> Variable(xmat[i,:]), 1:m)
                ys = softmax(scope, xs)
                @fact length(ys)             --> m
                @fact isa(ys, Vector)        --> true
                @fact eltype(ys) <: Constant --> true
                for i = 1:m
                    @fact size(ys[i]) --> (1, n)
                    @fact ys[i].data  --> roughly(map(FloatX, ymat[i,:]))
                end

                for j = 1:n
                    s = 0.0
                    for i = 1:m
                        s += ys[i].data[j]
                    end
                    @fact s --> roughly(1, stol)
                end
            end

            context("GradScope") do
                scope = GradScope()
                xmat = randn(m, n)
                ymat = softmax(xmat)
                xs = map(i -> Constant(xmat[i,:]), 1:m)
                ys = softmax(scope, xs)
                @fact length(ys)             --> m
                @fact isa(ys, Vector)        --> true
                @fact eltype(ys) <: Constant --> true
                for i = 1:m
                    @fact size(ys[i]) --> (1, n)
                    @fact ys[i].data  --> roughly(map(FloatX, ymat[i,:]))
                end

                for j = 1:n
                    s = 0.0
                    for i = 1:m
                        s += ys[i].data[j]
                    end
                    @fact s --> roughly(1, stol)
                end

                xmat = randn(m, n)
                ymat = softmax(xmat)
                xs = map(i -> Variable(xmat[i,:]), 1:m)
                ys = softmax(scope, xs)
                @fact isa(ys, Vector)        --> true
                @fact length(ys)             --> m
                @fact eltype(ys) <: Variable --> true
                for i = 1:m
                    @fact size(ys[i]) --> (1, n)
                    @fact ys[i].data  --> roughly(map(FloatX, ymat[i,:]))
                end

                for j = 1:n
                    s = 0.0
                    for i = 1:m
                        s += ys[i].data[j]
                    end
                    @fact s --> roughly(1, stol)
                end
            end

            context("Gradient") do
                xmat = randn(m, n)
                xs = map(i -> Variable(xmat[i,:]), 1:m)
                test_op_grad_mse((s, xs) -> concat(s, softmax(s, xs)), xs, wrt=xs)
            end
        end
    end
end
