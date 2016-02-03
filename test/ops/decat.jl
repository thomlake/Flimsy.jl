using Flimsy

facts("decat") do
    for (m, n) in [(6, 1), (3, 9)]
        context("$(m)x$(n)") do
            context("DataVariable") do
                scope = DynamicScope()
                gradscope = GradScope(scope)

                x = DataVariable(randn(m, n))
                y = decat(scope, x)
                @fact isa(y, Vector)            --> true
                @fact eltype(y) <: DataVariable --> true
                @fact length(y)                 --> m
                for i = 1:m
                    @fact size(y[i])       --> (1, n)
                    for j = 1:n
                        @fact y[i].data[j] --> roughly(x.data[i,j])
                    end
                end

                x = DataVariable(randn(m, n))
                y = decat(gradscope, x)
                @fact isa(y, Vector)            --> true
                @fact eltype(y) <: DataVariable --> true
                @fact length(y)                 --> m
                for i = 1:m
                    @fact size(y[i])       --> (1, n)
                    for j = 1:n
                        @fact y[i].data[j] --> roughly(x.data[i,j])
                    end
                end
            end

            context("GradVariable") do
                scope = DynamicScope()
                gradscope = GradScope(scope)

                x = GradVariable(randn(m, n), zeros(m, n))
                y = decat(scope, x)
                @fact isa(y, Vector)            --> true
                @fact eltype(y) <: DataVariable --> true
                @fact length(y)                 --> m
                for i = 1:m
                    @fact size(y[i])       --> (1, n)
                    for j = 1:n
                        @fact y[i].data[j] --> roughly(x.data[i,j])
                    end
                end

                x = GradVariable(randn(m, n), zeros(m, n))
                y = decat(gradscope, x)
                @fact isa(y, Vector)            --> true
                @fact eltype(y) <: GradVariable --> true
                @fact length(y)                 --> m
                for i = 1:m
                    @fact size(y[i])       --> (1, n)
                    for j = 1:n
                        @fact y[i].data[j] --> roughly(x.data[i,j])
                    end
                end
            end

            context("Gradient") do
                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse((s, x) -> concat(s, decat(s, x)), x, wrt=x)
            end
        end
    end
end
