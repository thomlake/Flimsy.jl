using Flimsy
using Flimsy.Components
using RDatasets
using MLBase

function check()
    # Example of how to apply finite difference
    # gradient checking to a Learner.
    n_sample = 2
    n_feature = 10
    n_labels = 3
    X = randn(n_feature, n_sample)
    Y = rand(1:n_labels, n_sample)
    theta = LogisticRegression(n_labels, n_feature)
    g() = gradient!(probs, theta, X, Y)
    c() = probs(theta, X, Y)[1]
    gradcheck(g, c, theta)
end

function demo()
    srand(123)

    df = dataset("datasets", "iris")
    response = [:Species]
    explanatory = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]

    features = Flimsy.Extras.zscore(convert(Array{Float64}, df[:, explanatory]).')
    labelstrings = vec(convert(Array{ASCIIString}, df[:, response]))
    lmap = labelmap(labelstrings)
    labels = labelencode(lmap, labelstrings)

    n_sample = size(features, 2)
    n_feature = length(explanatory)
    n_labels = length(lmap)

    trainmask = rand(n_sample) .< 0.75
    X_tr = features[:,trainmask]
    Y_tr = labels[trainmask]
    X_te = features[:,!trainmask]
    Y_te = labels[!trainmask]
    n_train = length(Y_tr)
    n_test = length(Y_te)

    println("[Info]")
    println("  number of features      => ", n_feature)
    println("  number of labels        => ", n_labels)
    println("  number of samples       => ", n_sample)
    println("  number of train samples => ", n_train)
    println("  number of test samples  => ", n_test)

    theta = LogisticRegression(n_labels, n_feature)

    sgd = optimizer(SGD, theta, learning_rate=0.01)

    progress = Flimsy.Extras.Progress(theta, patience=1, max_epochs=200) do
        return probs(theta, X_tr, Y_tr)[1] / n_test
    end

    # Fit parameters
    start(progress)
    while !quit(progress)
        gradient!(probs, theta, X_tr, Y_tr)
        update!(sgd, theta)
        step(progress)
    end
    done(progress)

    Yhat_tr = predict(theta, X_tr)
    Yhat_te = predict(theta, X_te)

    println("[Results]")
    println("  number of epochs => ", progress.epoch)
    println("  cpu time         => ", round(time(progress), 2), " seconds")
    println("  final train cost => ", progress.current_value)
    println("  train error      => ", sum(Y_tr .!= Yhat_tr) / n_train)
    println("  test error       => ", sum(Y_te .!= Yhat_te) / n_test)
end

check()
demo()
