# Flimsy.jl
# Sigmoid Regression

using Flimsy
using Flimsy.Components
import Flimsy.Demo: MoG

function create_data(D::Vector{MoG}, n::Int)
    X = Matrix{Float64}[]
    Y = Vector{Int}[]
    for i = 1:length(D)
        Xi, Yi = rand(D[i], n)
        push!(X, Xi)
        push!(Y, Yi)
    end
    return vcat(X...), hcat(Y...).' .== 2
end

function check()
    n_classes, n_features = 2, [5, 3]
    n_samples = 5
    X, Y = create_data([MoG(2, n_features[i]) for i = 1:n_classes], n_samples)
    params = SigmoidRegression(n_classes, sum(n_features))
    scope = Scope()
    g = () -> gradient!(cost, scope, params, Input(X), Y)
    c = () -> cost(scope, params, Input(X), Y)
    check_gradients(g, c, params)
end

function demo()
    n_classes, n_features = 2, [5, 3]
    n_train, n_test = 50, 50
    D = [MoG(2, n_features[i]) for i = 1:n_classes]
    X_train, Y_train = create_data(D, n_train)
    X_test, Y_test = create_data(D, n_test)

    scope = Scope()
    params = SigmoidRegression(n_classes, sum(n_features))
    opt = optimizer(GradientDescent, params, learning_rate=0.1)
    nll = Inf
    for epoch = 1:100
        nll_curr = gradient!(cost, scope, params, Input(X_train), Y_train)[1]
        update!(opt)
        @assert nll_curr <= nll
        nll = nll_curr
        epoch % 10 == 0 && println("[$epoch] nll => $nll")
    end
    println("err => ", sum(Y_test .!= predict(reset!(scope), params, Input(X_test))) / n_test)
end

("-c" in ARGS || "--check" in ARGS) && check()
demo()
