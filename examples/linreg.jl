# Linear Regression example using Nimble.jl package
# author: tllake
# email: thom.l.lake@gmail.com
#
# This example is for demonstration purposes only.
# Using gradient descent to fit a linear regression
# model on a dataset like Boston Housing is a poor life choice.

using RDatasets
using Flimsy
using Flimsy.Components

function check()
    # Example of how to apply finite difference
    # gradient checking to a Learner.
    n_samples = 2
    n_features = 10
    n_outputs = 3
    X = randn(n_features, n_samples)
    Y = randn(n_outputs, n_samples)
    theta = LinearRegression(n_outputs, n_features)
    g() = gradient!(cost, theta, X, Y)
    c() = cost(theta, X, Y)
    gradcheck(g, c, theta)
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
    features = Flimsy.Extras.zscore(convert(Array{Float64}, df[:, expl]).')
    targets = Flimsy.Extras.zscore(convert(Array{Float64}, df[:, resp]).')

    # Build the model
    n_samples = size(features, 2)
    n_features = length(expl)
    n_outputs = length(resp)
    theta = LinearRegression(n_outputs, n_features)

    # Use the Progress type to assess training convergence.
    progress = Flimsy.Extras.Progress(theta, max_epochs=Inf, patience=0, tol=1e-10)

    # Update type
    opt = optimizer(GradientDescent, theta, learning_rate=0.01 / n_samples)

    # Main training loop.
    start(progress)
    while !quit(progress)
        nll = gradient!(cost, theta, features, targets)
        update!(opt, theta)
        step(progress, nll)
    end
    done(progress)

    # Print out info about our model.
    println("[Info]")
    println("  number of samples: ", n_samples)
    println("  number of features: ", n_features)
    println("  converged after ", progress.epoch, " epochs (", round(time(progress), 2), " seconds)")
    println("  nll: ",  progress.best_value)
    println("[Coefficients]")
    for (k, v) in zip(expl, theta.w.data)
        println("  ", rpad(k, 7), " => ", sign(v) > 0 ? "+" : "-", round(abs(v), 3))
    end
end

check()
demo()
