using Flimsy


facts("softmax_vector") do
    context("Mx1x1") do
        m, n = 8, 1
        context("Scope") do
            scope = DynamicScope()
            xmat = randn(m, n)
            ymat = softmax(xmat)
            xs = [DataVariable(xmat[i,:]) for i = 1:m]
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
            xs = [GradVariable(xmat[i,:]) for i = 1:m]
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
            xs = [DataVariable(xmat[i,:]) for i = 1:m]
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
            xs = [GradVariable(xmat[i,:]) for i = 1:m]
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
            xs = [GradVariable(xmat[i,:]) for i = 1:m]
            f(s, xs) = concat(s, softmax(s, xs))
            f(xs) = concat(softmax(xs))
            test_op_grad_mse(f, xs, wrt=xs)
        end
    end


    context("Mx1xN") do
        m, n = 5, 8
        context("Scope") do
            scope = DynamicScope()
            xmat = randn(m, n)
            ymat = softmax(xmat)
            xs = [DataVariable(xmat[i,:]) for i = 1:m]
            ys = softmax(scope, xs)
            @fact isa(ys, Vector)            --> true
            @fact length(ys)                 --> m
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
            xs = [GradVariable(xmat[i,:]) for i = 1:m]
            ys = softmax(scope, xs)
            @fact isa(ys, Vector)            --> true
            @fact length(ys)                 --> m
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
            xs = DataVariable[DataVariable(xmat[i,:]) for i = 1:m]
            ys = softmax(scope, xs)
            @fact isa(ys, Vector)            --> true
            @fact length(ys)                 --> m
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
            xs = [GradVariable(xmat[i,:]) for i = 1:m]
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

            context("Gradient") do
                xmat = randn(m, n)
                xs = [GradVariable(xmat[i,:]) for i = 1:m]
                f(s, xs) = concat(s, softmax(s, xs))
                f(xs) = concat(softmax(xs))
                test_op_grad_mse(f, xs, wrt=xs)
            end
        end
    end
end

