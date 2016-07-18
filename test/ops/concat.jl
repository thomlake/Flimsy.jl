using Flimsy

facts("concat") do
    m_min, m_max = 2, 7
    for K in [3, 4, 5]
        for n in [1, 3]
            context("$(K)xMx$(n)") do
                context("Constant") do
                    ms = rand(m_min:m_max, K)
                    m = sum(ms)
                    xs = map(k -> Constant(randn(ms[k], n)), 1:K)
                    y = concat(RunScope(), xs)
                    @fact isa(y, Constant) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> vcat(map(x -> x.data, xs)...)

                    y = concat(GradScope(), xs)
                    @fact isa(y, Constant) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> vcat(map(x -> x.data, xs)...)
                end
                
                context("Variable") do
                    ms = rand(m_min:m_max, K)
                    m = sum(ms)
                    xs = map(k -> Variable(randn(ms[k], n)), 1:K)
                    y = concat(RunScope(), xs)
                    @fact isa(y, Constant) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> vcat(map(x -> x.data, xs)...)

                    y = concat(GradScope(), xs)
                    @fact isa(y, Variable) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> vcat(map(x -> x.data, xs)...)
                end

                context("Mixed") do
                    ms = rand(m_min:m_max, K)
                    m = sum(ms)
                    xs = AbstractValue[Constant(randn(ms[1], n))]
                    for k = 2:K 
                        push!(xs, Variable(randn(ms[k], n))) 
                    end
                    y = concat(RunScope(), xs)
                    @fact isa(y, Constant) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> vcat(map(x -> x.data, xs)...)

                    y = concat(GradScope(), xs)
                    @fact isa(y, Variable) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> vcat(map(x -> x.data, xs)...)
                end
                
                context("Gradient") do
                    ms = rand(m_min:m_max, K)
                    xs = map(k -> Variable(randn(ms[k], n)), 1:K)
                    test_op_grad_mse(concat, xs, wrt=xs)
                end
            end
        end
    end
end
