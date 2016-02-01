
facts("tanh") do
    context("Mx1") do
        m, n = 3, 1
        context("Scope") do
            scope = DynamicScope()
            x = DataVariable(randn(m))
            y = tanh(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y) --> (m, n) 
            @fact y.data --> roughly(tanh(x.data))

            x = GradVariable(randn(m))
            y = tanh(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> roughly(tanh(x.data))
        end

        context("GradScope") do
            scope = GradScope(DynamicScope())
            x = DataVariable(randn(m))
            y = tanh(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> roughly(tanh(x.data))

            x = GradVariable(randn(m))
            y = tanh(scope, x)
            @fact isa(y, GradVariable) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> roughly(tanh(x.data))
        end

        context("Gradient") do
            x = GradVariable(randn(m))
            test_op_grad_mse(tanh, x, wrt=x)
        end
    end

    context("MxN") do
        m, n = 5, 8
        context("Scope") do
            scope = DynamicScope()
            x = DataVariable(randn(m, n))
            y = tanh(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> roughly(tanh(x.data))

            x = GradVariable(randn(m, n))
            y = tanh(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> roughly(tanh(x.data))
        end

        context("GradScope") do
            scope = GradScope(DynamicScope())
            x = DataVariable(randn(m, n))
            y = tanh(scope, x)
            @fact isa(y, DataVariable) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> roughly(tanh(x.data))

            x = GradVariable(randn(m, n))
            y = tanh(scope, x)
            @fact isa(y, GradVariable) --> true
            @fact size(y) --> (m, n)
            @fact y.data --> roughly(tanh(x.data))
        end

        x = GradVariable(randn(m, n))
        test_op_grad_mse(tanh, x, wrt=x)
    end
end