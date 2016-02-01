using Flimsy

facts("softmax") do
    context("Mx1") do
        m, n = 5, 1
        context("Array") do
            x = randn(m)
            y = softmax(x)
            @fact isa(y, Vector) --> true
            @fact size(y)        --> (m,)
            @fact y              --> roughly(exp(x) / sum(exp(x)))
            @fact sum(y)         --> roughly(1)
        end

        context("Scope") do
            scope = DynamicScope()
            
            x = DataVariable(randn(m))
            y = softmax(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y)              --> (m, n)
            @fact y.data               --> roughly(softmax(x.data))
            @fact sum(y.data)          --> roughly(1)
            
            x = GradVariable(randn(m))
            y = softmax(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y)              --> (m, n)
            @fact y.data               --> roughly(softmax(x.data))
            @fact sum(y.data)          --> roughly(1)
        end

        context("GradScope") do
            scope = GradScope(DynamicScope())

            x = DataVariable(randn(m))
            y = softmax(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y)              --> (m, n)
            @fact y.data               --> roughly(softmax(x.data))
            @fact sum(y.data)          --> roughly(1)

            x = GradVariable(randn(m))
            y = softmax(scope, x)
            @fact isa(y, GradVariable) --> true
            @fact size(y)              --> (m, n)
            @fact y.data               --> roughly(softmax(x.data))
            @fact sum(y.data)          --> roughly(1)
        end

        context("Gradient") do
            x = GradVariable(randn(m))
            test_op_grad_mse(softmax, x, wrt=x)
        end
    end
    

    context("MxN") do
        m, n = 4, 7
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

            x = GradVariable(randn(m, n))
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

            x = GradVariable(randn(m, n))
            y = softmax(scope, x)
            @fact isa(y, GradVariable) --> true
            @fact size(y)              --> (m, n)
            @fact y.data               --> roughly(softmax(x.data))
            @fact sum(y.data, 1)       --> roughly(ones(1, n))
            @fact sum(y.data)          --> roughly(n)
        end

        context("Gradient") do
            x = GradVariable(randn(m, n))
            test_op_grad_mse(softmax, x, wrt=x)
        end
    end
end
