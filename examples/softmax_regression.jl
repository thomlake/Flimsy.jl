# Flimsy.jl
# Logistic Regression

using Flimsy
using Flimsy.Components
using RDatasets
using MLBase

function check()
    n_sample = 2
    n_labels, n_feature = 3, 10
    X = randn(n_feature, n_sample)
    Y = rand(1:n_labels, n_sample)

    model = Model(
        SoftmaxRegression,
        w=rand(Normal(0, 0.01), n_labels, n_feature),
        b=zeros(n_labels),
    )
    gcost = compile(:cost, model, Variable, Int, gradients=true)
    cost = compile(:cost, model, Variable, Int, gradients=false)
    # c = F.compile(cost, params, typeof(X), typeof(Y), gradients=false)
    g = () -> gradient!(cost, params, GradInput(X), Y)
    c = () -> cost(params, Input(X), Y)
    check_gradients(g, c, gparams)
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
    opt = optimizer(GradientDescent, theta, learning_rate=0.01)
    progress = Progress(theta, patience=1, max_epochs=200)

    # Fit parameters
    start(progress)
    while !quit(progress)
        nll = gradient!(cost, theta, X_tr, Y_tr)
        update!(opt, theta)
        step(progress, nll)
    end
    done(progress)

    Yhat_tr = predict(theta, X_tr)
    Yhat_te = predict(theta, X_te)

    println("[Results]")
    println("  number of epochs => ", progress.epoch)
    println("  cpu time         => ", round(time(progress), 2), " seconds")
    println("  final train nll  => ", progress.current_value)
    println("  train error      => ", sum(Y_tr .!= Yhat_tr) / n_train)
    println("  test error       => ", sum(Y_te .!= Yhat_te) / n_test)
end

check()
# demo()