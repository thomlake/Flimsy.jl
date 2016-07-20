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
    params = SigmoidRegression(n_classes, sum(n_features))
    check_gradients(cost, params, Input(X), Y)
end

function demo()
    srand(1234)
    feature_vec_sizes = [5, 3]
    n_classes = length(feature_vec_sizes)
    n_features = sum(feature_vec_sizes)
    n_train, n_test = 50, 50
    X, Y = create_data(feature_vec_sizes, n_train + n_test)
    @assert size(X) == (n_features, n_train + n_test)
    @assert size(Y) == (n_classes, n_train + n_test)
    X_train, X_test = X[:,1:n_train], X[:,n_train+1:end]
    Y_train, Y_test = Y[:,1:n_train], Y[:,n_train+1:end]

    params = SigmoidRegression(n_classes, n_features)
    opt = optimizer(GradientDescent, params, learning_rate=0.1)
    nll = Inf
    start_time = time()
    for epoch = 1:500
        nll_curr = @backprop cost(params, Input(X_train), Y_train)
        update!(opt)
        nll_curr <= nll || error(nll_curr, " > ", nll)
        nll = nll_curr
        epoch % 50 == 0 && println("[$epoch] nll => $nll")
    end
    stop_time = time()
    Y_pred = @run predict(params, Input(X_test))
    println("wall time  => ", stop_time - start_time)
    println("test error => ", sum(Y_test .!= Y_pred) / n_test)
end

("-c" in ARGS || "--check" in ARGS) && check()
demo()
