using Flimsy

function test_softmax()
const stol = 1e-5
    facts("softmax") do
        for (m, n) in [(3, 1), (4, 7)]
            context("$(m)x$(n)") do

                context("Array") do
                    x = randn(m, n)
                    y = softmax(x)
                    @fact isa(y, Matrix) --> true
                    @fact size(y)        --> (m, n)
                    @fact y              --> roughly(exp(x) ./ sum(exp(x), 1))
                    @fact sum(y, 1)      --> roughly(ones(1, n), stol)
                    @fact sum(y)         --> roughly(n)
                end

                context("RunScope") do
                    scope = RunScope()
                    x = Constant(randn(m, n))
                    y = softmax(scope, x)
                    @fact isa(y, Constant) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> roughly(softmax(x.data))
                    @fact sum(y.data, 1)   --> roughly(ones(1, n), stol)
                    @fact sum(y.data)      --> roughly(n, stol)

                    x = Variable(randn(m, n))
                    y = softmax(scope, x)
                    @fact isa(y, Constant) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> roughly(softmax(x.data))
                    @fact sum(y.data, 1)   --> roughly(ones(1, n), stol)
                    @fact sum(y.data)      --> roughly(n, stol)
                end

                context("GradScope") do
                    scope = GradScope()
                    x = Constant(randn(m, n))
                    y = softmax(scope, x)
                    @fact isa(y, Constant) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> roughly(softmax(x.data))
                    @fact sum(y.data, 1)   --> roughly(ones(1, n), stol)
                    @fact sum(y.data)      --> roughly(n, stol)

                    x = Variable(randn(m, n))
                    y = softmax(scope, x)
                    @fact isa(y, Variable) --> true
                    @fact size(y)          --> (m, n)
                    @fact y.data           --> roughly(softmax(x.data))
                    @fact sum(y.data, 1)   --> roughly(ones(1, n), stol)
                    @fact sum(y.data)      --> roughly(n, stol)
                end

                context("Gradient") do
                    x = Variable(randn(m, n))
                    test_op_grad_mse(softmax, x, wrt=x)
                end
            end
        end
    end
end
test_softmax()
