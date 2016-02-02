using Flimsy

facts("softmax") do
    for (m, n) in [(3, 1), (4, 7)]
        context("$(m)x$(n)") do
            context("Array") do
                x = randn(m, n)
                y = softmax(x)
                @fact isa(y, Matrix) --> true
                @fact size(y)        --> (m, n)
                @fact y              --> roughly(exp(x) ./ sum(exp(x), 1))
                @fact sum(y, 1)      --> roughly(ones(1, n))
                @fact sum(y)         --> roughly(n)
            end

            context("Scope") do
                scope = DynamicScope()
                x = DataVariable(randn(m, n))
                y = softmax(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(softmax(x.data))
                @fact sum(y.data, 1)       --> roughly(ones(1, n))
                @fact sum(y.data)          --> roughly(n)

                x = GradVariable(randn(m, n), zeros(m, n))
                y = softmax(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(softmax(x.data))
                @fact sum(y.data, 1)       --> roughly(ones(1, n))
                @fact sum(y.data)          --> roughly(n)
            end

            context("GradScope") do
                scope = GradScope(DynamicScope())
                x = DataVariable(randn(m, n))
                y = softmax(scope, x)
                @fact isa(y, DataVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(softmax(x.data))
                @fact sum(y.data, 1)       --> roughly(ones(1, n))
                @fact sum(y.data)          --> roughly(n)

                x = GradVariable(randn(m, n), zeros(m, n))
                y = softmax(scope, x)
                @fact isa(y, GradVariable) --> true
                @fact size(y)              --> (m, n)
                @fact y.data               --> roughly(softmax(x.data))
                @fact sum(y.data, 1)       --> roughly(ones(1, n))
                @fact sum(y.data)          --> roughly(n)
            end

            context("Gradient") do
                x = GradVariable(randn(m, n), zeros(m, n))
                test_op_grad_mse(softmax, x, wrt=x)
            end
        end
    end
end
