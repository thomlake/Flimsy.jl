# Flimsy.jl
# Linear Regression

# This example is for demonstration purposes only.
# Using gradient descent to fit a linear regression
# model on a dataset like Boston Housing is a poor life choice.
using Flimsy
using Flimsy.Components
import Flimsy.Extras: zscore
import RDatasets: dataset

# Check computed gradients using finite differences
function check()
    println("checking gradients....")
    n_samples = 2
    n_targets, n_features = 3, 10
    X = randn(n_features, n_samples)
    Y = randn(n_targets, n_samples)
    params = LinearRegression(n_targets, n_features)
    check_gradients(cost, params, Input(X), Y)
end

function main()
    srand(123)

    # Load some data.
    df = dataset("MASS", "boston")
    resp = [:MedV]
    expl = [:Rm, :Crim, :LStat, :PTRatio, :Dis]

    # Convert data frame to Arrays and preprocess to have
    # zero mean and unit standard deviation. 
    # N.B. Julia matrices are column-major, 
    # so we transpose the features and targets matrices.
    # After this features[i,j] is the ith feature of the jth instance.
    features = zscore(convert(Array{Float64}, df[:, expl]).')
    targets = zscore(convert(Array{Float64}, df[:, resp]).')

    # Build the model
    n_samples = size(features, 2)
    n_features = length(expl)
    n_targets = length(resp)
    params = LinearRegression(n_targets, n_features)

    # Learning Algorithm
    opt = optimizer(GradientDescent, params, learning_rate=0.01 / n_samples)

    # Main training loop
    ds = DataScope()
    gs = GradScope()

    nll_prev, nll = Inf, -Inf
    n_epochs = 0
    start_time = time()
    while true
        n_epochs += 1
        nll = cost(gs, params, Input(features), targets)
        backprop!(gs)
        update!(opt)
        nll_prev - nll > 1e-6 || break
        nll_prev = nll
    end
    stop_time = time()

    # Print info about our model.
    println("[Info]")
    println("  number of samples  => ", n_samples)
    println("  number of features => ", n_features)
    println("  number of epochs   => ", n_epochs)
    println("  cpu time           => ", round(stop_time - start_time, 2), " seconds")
    println("  final train nll    => ", nll)
    println("[Coefficients]")
    for (k, v) in zip(expl, params.w.data)
        println("  ", rpad(k, 7), " => ", sign(v) > 0 ? "+" : "-", round(abs(v), 3))
    end
end

("-c" in ARGS || "--check" in ARGS) && check()
main()
