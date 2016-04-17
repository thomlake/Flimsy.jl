using Flimsy

facts("concat") do
    m_min, m_max = 2, 7
    for K in [3, 4, 5]
        for n in [1, 3]
            context("$(K)xMx$(n)") do
                context("DataVariable") do
                    scope = DataScope()
                    gradscope = GradScope()

                    ms = rand(m_min:m_max, K)
                    m = sum(ms)
                    xs = map(k -> DataVariable(randn(ms[k], n)), 1:K)
                    y = concat(scope, xs)
                    @fact isa(y, DataVariable) --> true
                    @fact size(y)              --> (m, n)
                    @fact y.data               --> vcat(map(x -> x.data, xs)...)

                    y = concat(gradscope, xs)
                    @fact isa(y, DataVariable) --> true
                    @fact size(y)              --> (m, n)
                    @fact y.data               --> vcat(map(x -> x.data, xs)...)
                end
                
                context("GradVariable") do
                    scope = DataScope()
                    gradscope = GradScope()

                    ms = rand(m_min:m_max, K)
                    m = sum(ms)
                    xs = map(k -> GradVariable(randn(ms[k], n), zeros(ms[k], n)), 1:K)
                    y = concat(scope, xs)
                    @fact isa(y, DataVariable) --> true
                    @fact size(y)              --> (m, n)
                    @fact y.data               --> vcat(map(x -> x.data, xs)...)

                    y = concat(gradscope, xs)
                    @fact isa(y, GradVariable) --> true
                    @fact size(y)              --> (m, n)
                    @fact y.data               --> vcat(map(x -> x.data, xs)...)

                end
                
                context("Gradient") do
                    ms = rand(m_min:m_max, K)
                    xs = map(k -> GradVariable(randn(ms[k], n), zeros(ms[k], n)), 1:K)
                    test_op_grad_mse(concat, xs, wrt=xs)
                end
            end
        end
    end
end
