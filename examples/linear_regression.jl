# Flimsy.jl
# Linear Regression

# This example is for demonstration purposes only.
# Using gradient descent to fit a linear regression
# model on a dataset like Boston Housing is a poor life choice.
using Flimsy
using Flimsy.Components
import Flimsy.Extras: zscore
import RDatasets: dataset

function check()
    # check computed gradients using finite differences
    println("checking gradients....")
    n_samples = 2
    n_targets, n_features = 3, 10
    X = randn(n_features, n_samples)
    Y = randn(n_targets, n_samples)
    params = LinearRegression(n_targets, n_features)
    scope = Scope()
    g() = gradient!(cost, scope, params, Input(X), Y)
    c() = (reset!(scope); cost(scope, params, Input(X), Y))
    check_gradients(g, c, params)
end

function demo()
    srand(123)

    # Load some data.
    df = dataset("MASS", "boston")
    resp = [:MedV]
    expl = [:Rm, :Crim, :LStat, :PTRatio, :Dis]

    # Convert data frame to Arrays and preprocess to have
    # zero mean and unit stard deviation. Notice the arrays
    # are transposed so columns are instances and rows are features.
    features = zscore(convert(Array{Float64}, df[:, expl]).')
    targets = zscore(convert(Array{Float64}, df[:, resp]).')

    # Build the model
    n_samples = size(features, 2)
    n_features = length(expl)
    n_targets = length(resp)
    params = LinearRegression(n_targets, n_features)
    scope = Scope()

    # Learning Algorithm
    opt = optimizer(GradientDescent, params, learning_rate=0.01 / n_samples)

    # Use the Progress type to monitor training convergence.
    progress = Progress(params, ExternalEvaluation(tol=0.001), NoImprovement(), max_epochs=Inf)

    # Main training loop.
    while !converged(progress)
        nll = gradient!(cost, scope, params, Input(features), targets)
        update!(opt)
        progress(nll)
    end
    timer_stop(progress)

    # Print out info about our model.
    println("[Info]")
    println("  number of samples  => ", n_samples)
    println("  number of features => ", n_features)
    println("  number of epochs   => ", epoch(progress))
    println("  cpu time           => ", round(time(progress), 2), " seconds")
    println("  final train nll    => ", evaluate(progress, best=true))
    println("[Coefficients]")
    for (k, v) in zip(expl, params.w.data)
        println("  ", rpad(k, 7), " => ", sign(v) > 0 ? "+" : "-", round(abs(v), 3))
    end
end

("-c" in ARGS || "--check" in ARGS) && check()
demo()
