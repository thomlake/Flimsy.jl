# Flimsy.jl
# Logistic Regression

using Flimsy
using Flimsy.Components
import Flimsy.Extras: zscore
import RDatasets: dataset
import MLBase: labelmap, labelencode

Params(n_labels, n_features) = LogisticRegression(
    w=rand(Normal(0, 0.01), n_labels, n_features),
    b=zeros(n_labels),
)

function check()
    n_sample = 2
    n_labels, n_features = 3, 10
    X = randn(n_features, n_sample)
    Y = rand(1:n_labels, n_sample)
    params = Params(n_labels, n_features)
    g = () -> gradient!(cost, params, Input(X), Y)
    c = () -> cost(params, Input(X), Y)
    check_gradients(g, c, params)
end

function demo()
    srand(123)

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
    X_tr = features[:,trainmask]
    Y_tr = labels[trainmask]
    X_te = features[:,!trainmask]
    Y_te = labels[!trainmask]
    n_train = length(Y_tr)
    n_test = length(Y_te)

    println("[Info]")
    println("  number of features      => ", n_features)
    println("  number of labels        => ", n_labels)
    println("  number of samples       => ", n_sample)
    println("  number of train samples => ", n_train)
    println("  number of test samples  => ", n_test)
    params = Params(n_labels, n_features)
    opt = optimizer(GradientDescent, params, learning_rate=0.01)
    progress = Flimsy.Progress(params, patience=1, max_epochs=200)

    # Fit parameters
    start(progress)
    while !quit(progress)
        nll = gradient!(cost, params, Input(X_tr), Y_tr)
        update!(opt, params)
        step(progress, nll)
    end
    done(progress)

    Yhat_tr = predict(params, Input(X_tr))
    Yhat_te = predict(params, Input(X_te))

    println("[Results]")
    println("  number of epochs => ", progress.epoch)
    println("  cpu time         => ", round(time(progress), 2), " seconds")
    println("  final train nll  => ", progress.current_value)
    println("  train error      => ", sum(Y_tr .!= Yhat_tr) / n_train)
    println("  test error       => ", sum(Y_te .!= Yhat_te) / n_test)
end

check()
demo()
