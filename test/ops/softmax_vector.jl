using Flimsy

facts("softmax_vector") do
    for (m, n) in [(8, 1), (5, 8)]
        context("$(m)x1x$(n)") do
            context("Scope") do
                scope = DynamicScope()
                xmat = randn(m, n)
                ymat = softmax(xmat)
                xs = map(i -> DataVariable(xmat[i,:]), 1:m)
                ys = softmax(scope, xs)
                @fact length(ys)                 --> m
                @fact isa(ys, Vector)            --> true
                @fact eltype(ys) <: DataVariable --> true
                for i = 1:m
                    @fact size(ys[i]) --> (1, n)
                    @fact ys[i].data  --> roughly(ymat[i,:])
                end

                for j = 1:n
                    s = 0.0
                    for i = 1:m
                        s += ys[i].data[j]
                    end
                    @fact s --> roughly(1)
                end

                xmat = randn(m, n)
                ymat = softmax(xmat)
                xs = map(i -> GradVariable(xmat[i,:], zeros(1, n)), 1:m)
                ys = softmax(scope, xs)
                @fact length(ys)                 --> m
                @fact isa(ys, Vector)            --> true
                @fact eltype(ys) <: DataVariable --> true
                for i = 1:m
                    @fact size(ys[i]) --> (1, n)
                    @fact ys[i].data  --> roughly(ymat[i,:])
                end

                for j = 1:n
                    s = 0.0
                    for i = 1:m
                        s += ys[i].data[j]
                    end
                    @fact s --> roughly(1)
                end
            end

            context("GradScope") do
                scope = GradScope(DynamicScope())
                xmat = randn(m, n)
                ymat = softmax(xmat)
                xs = map(i -> DataVariable(xmat[i,:]), 1:m)
                ys = softmax(scope, xs)
                @fact length(ys)                 --> m
                @fact isa(ys, Vector)            --> true
                @fact eltype(ys) <: DataVariable --> true
                for i = 1:m
                    @fact size(ys[i]) --> (1, n)
                    @fact ys[i].data  --> roughly(ymat[i,:])
                end

                for j = 1:n
                    s = 0.0
                    for i = 1:m
                        s += ys[i].data[j]
                    end
                    @fact s --> roughly(1)
                end

                xmat = randn(m, n)
                ymat = softmax(xmat)
                xs = map(i -> GradVariable(xmat[i,:], zeros(1, n)), 1:m)
                ys = softmax(scope, xs)
                @fact isa(ys, Vector)            --> true
                @fact length(ys)                 --> m
                @fact eltype(ys) <: GradVariable --> true
                for i = 1:m
                    @fact size(ys[i]) --> (1, n)
                    @fact ys[i].data  --> roughly(ymat[i,:])
                end

                for j = 1:n
                    s = 0.0
                    for i = 1:m
                        s += ys[i].data[j]
                    end
                    @fact s --> roughly(1)
                end
            end

            context("Gradient") do
                xmat = randn(m, n)
                xs = map(i -> GradVariable(xmat[i,:], zeros(1, n)), 1:m)
                test_op_grad_mse((s, xs) -> concat(s, softmax(s, xs)), xs, wrt=xs)
            end
        end
    end
end
