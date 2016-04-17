# Flimsy.jl
# Softmax/Logistic Regression

using Flimsy
using Flimsy.Components
import Flimsy.Extras: zscore
import RDatasets: dataset
import MLBase: labelmap, labelencode

function check()
    n_sample = 2
    n_labels, n_features = 3, 10
    X = randn(n_features, n_sample)
    y = rand(1:n_labels, n_sample)
    params = Runtime(SoftmaxRegression(n_labels, n_features))
    check_gradients(cost, params, Input(X), y)
end

function demo()
    srand(123)
    # Load and preprocess data
    df = dataset("datasets", "iris")
    response = [:Species]
    explanatory = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]

    features = zscore(convert(Array{Float64}, df[:, explanatory]).')
    labelstrings = vec(convert(Array{ASCIIString}, df[:, response]))
    lmap = labelmap(labelstrings)
    labels = labelencode(lmap, labelstrings)

    n_sample = size(features, 2)
    n_features = length(explanatory)
    n_labels = length(lmap)

    trainmask = rand(n_sample) .< 0.75
    X_train = features[:,trainmask]
    y_train = labels[trainmask]
    X_test = features[:,!trainmask]
    y_test = labels[!trainmask]
    n_train = length(y_train)
    n_test = length(y_test)

    println("[Info]")
    println("  number of features      => ", n_features)
    println("  number of labels        => ", n_labels)
    println("  number of samples       => ", n_sample)
    println("  number of train samples => ", n_train)
    println("  number of test samples  => ", n_test)

    # Setup parameters and create optimizer
    params = Runtime(SoftmaxRegression(n_labels, n_features))
    opt = optimizer(GradientDescent, params, learning_rate=0.01)
    
    # Main training loop
    nll_prev, nll = Inf, -Inf
    max_epochs, n_epochs = 100, 0
    start_time = time()
    while n_epochs < max_epochs
        n_epochs += 1
        nll = cost(params, Input(X_train), y_train; grad=true)
        update!(opt)
        nll_prev - nll > 1e-6 || break
        nll_prev = nll
    end
    stop_time = time()

    p_train = predict(params, Input(X_train))
    p_test = predict(params, Input(X_test))

    println("[Results]")
    println("  number of epochs => ", n_epochs)
    println("  cpu time         => ", round(stop_time - start_time, 2), " seconds")
    println("  final train nll  => ", nll)
    println("  train error      => ", sum(y_train .!= p_train) / n_train)
    println("  test error       => ", sum(y_test .!= p_test) / n_test)
end

("-c" in ARGS || "--check" in ARGS) && check()
demo()
