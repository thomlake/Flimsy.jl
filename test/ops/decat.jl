using Flimsy

facts("decat") do
    for (m, n) in [(6, 1), (3, 9)]
        context("$(m)x$(n)") do
            context("Constant") do
                x = Constant(randn(m, n))
                y = decat(RunScope(), x)
                @fact isa(y, Vector) --> true
                @fact eltype(y) <: Constant --> true
                @fact length(y) --> m
                for i = 1:m
                    @fact size(y[i]) --> (1, n)
                    for j = 1:n
                        @fact y[i].data[j] --> roughly(x.data[i,j])
                    end
                end

                x = Constant(randn(m, n))
                y = decat(GradScope(), x)
                @fact isa(y, Vector) --> true
                @fact eltype(y) <: Constant --> true
                @fact length(y) --> m
                for i = 1:m
                    @fact size(y[i]) --> (1, n)
                    for j = 1:n
                        @fact y[i].data[j] --> roughly(x.data[i,j])
                    end
                end
            end

            context("Variable") do
                x = Variable(randn(m, n))
                y = decat(RunScope(), x)
                @fact isa(y, Vector) --> true
                @fact eltype(y) <: Constant --> true
                @fact length(y) --> m
                for i = 1:m
                    @fact size(y[i]) --> (1, n)
                    for j = 1:n
                        @fact y[i].data[j] --> roughly(x.data[i,j])
                    end
                end

                x = Variable(randn(m, n))
                y = decat(GradScope(), x)
                @fact isa(y, Vector) --> true
                @fact eltype(y) <: Variable --> true
                @fact length(y) --> m
                for i = 1:m
                    @fact size(y[i]) --> (1, n)
                    for j = 1:n
                        @fact y[i].data[j] --> roughly(x.data[i,j])
                    end
                end
            end

            context("Gradient") do
                x = Variable(randn(m, n))
                test_op_grad_mse((s, x) -> concat(s, decat(s, x)), x, wrt=x)
            end
        end
    end
end
