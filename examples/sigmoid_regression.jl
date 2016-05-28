# Flimsy.jl
# Sigmoid Regression
using Synthetic
using Flimsy
using Flimsy.Components

function create_data(feature_vec_sizes::Vector{Int}, n_samples::Int)
    mogs = [Synthetic.MixtureTask(d, 2) for d in feature_vec_sizes]
    X = Vector{Float64}[]
    Y = Vector{Int}[]
    for i = 1:n_samples
        x = Float64[]
        y = Int[]
        for mog in mogs
            xj, yj = rand(mog)
            append!(x, xj)
            push!(y, yj)
        end
        push!(X, x)
        push!(Y, y)
    end
    return hcat(X...), hcat(Y...) .== 2
end

function check()
    n_samples = 5
    feature_vec_sizes = [5, 3]
    n_classes = length(feature_vec_sizes)
    n_features = sum(feature_vec_sizes)
    X, Y = create_data(feature_vec_sizes, n_samples)
    params = Runtime(SigmoidRegression(n_classes, sum(n_features)))
    check_gradients(cost, params, Input(X), Y)
end

function demo()
    feature_vec_sizes = [5, 3]
    n_classes = length(feature_vec_sizes)
    n_features = sum(feature_vec_sizes)
    n_train, n_test = 50, 50
    X, Y = create_data(feature_vec_sizes, n_train + n_test)
    @assert size(X) == (n_features, n_train + n_test)
    @assert size(Y) == (n_classes, n_train + n_test)
    X_train, X_test = X[:,1:n_train], X[:,n_train+1:end]
    Y_train, Y_test = Y[:,1:n_train], Y[:,n_train+1:end]

    params = Runtime(SigmoidRegression(n_classes, n_features))
    opt = optimizer(GradientDescent, params, learning_rate=0.1)
    nll = Inf
    for epoch = 1:500
        nll_curr = cost(params, Input(X_train), Y_train, grad=true)
        update!(opt)
        @assert nll_curr <= nll
        nll = nll_curr
        epoch % 10 == 0 && println("[$epoch] nll => $nll")
    end
    println("err => ", sum(Y_test .!= predict(params, Input(X_test))) / n_test)
end

("-c" in ARGS || "--check" in ARGS) && check()
demo()
