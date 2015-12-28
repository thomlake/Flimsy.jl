# Flimsy.jl
# MNIST classification

using Flimsy
using Flimsy.Components
import Flimsy.Components: cost, predict
import MNIST

immutable Params <: Component
    output::LogisticRegression
    hidden::Vector{FeedForwardLayer}
end

@flimsy predict(theta::Params, x) = predict(theta.output, feedforward(theta.hidden, x))

@flimsy cost(theta::Params, x, y) = cost(theta.output, feedforward(theta.hidden, x), y)

function image_string(x::Vector, symbols::Vector{Char}=['-', '+'])
    s = isqrt(length(x))
    img = reshape(x, (s, s))
    rows = ASCIIString[]
    for i in 1:s
        push!(rows, join(symbols[int(img[i,:] .> 0.3) + 1]))
    end
    return join(rows, '\n')
end

function check()
    srand(123)
    rawX, rawY = MNIST.traindata()
    X = rawX[:,1]
    Y = round(Int, rawY + 1)[1]
    @assert all(isfinite(X))
    @assert all(isfinite(Y))
    hidden = multilayer(relu, 20, 40, 50, 60, 100, size(X, 1))
    output = LogisticRegression(10, 5)
    theta = Params(output, hidden)
    g() = gradient!(cost, theta, X, Y)
    c() = cost(theta, X, Y)
    gradcheck(g, c, theta)
end

function show_example_data()
    X, Y = MNIST.traindata()
    for k = 1:5
        i = rand(1:length(Y))
        println(int(Y[i]))
        println(image_string(X[:,i]))
    end
end

function fit()
    srand(123)
    rawX, rawY = MNIST.traindata()
    n_valid = max(10, round(Int, 0.1 * size(rawX, 2)))
    n_train = size(rawX, 2) - n_valid
    X_train, Y_train = rawX[:,1:n_train], round(Int, rawY[1:n_train] + 1)
    X_valid, Y_valid = rawX[:,n_train + 1:end], round(Int, rawY[n_train + 1:end] + 1)

    println("number train: $n_train")
    println("number valid: $n_valid")

    @assert all(isfinite(X_train))
    @assert all(isfinite(Y_train))
    @assert all(isfinite(X_valid))
    @assert all(isfinite(Y_valid))

    batch_size = 100
    indices = collect(1:n_train)

    hidden = multilayer(relu, 30, 40, 50, 60, 100, size(X_train, 1))
    output = LogisticRegression(10, 30)
    theta = Params(output, hidden)

    rmsprop = optimizer(RMSProp, theta, learning_rate=0.01 / batch_size, decay=0.8, clip=5.0, clipping_type=:scale)

    progress = Progress(theta, patience=0) do
        return sum(Y_valid .!= predict(theta, X_valid)) / n_valid
    end

    println("fitting...")
    start(progress)
    while !quit(progress)
        shuffle!(indices)
        for i = 1:batch_size:n_train
            idx = indices[i:min(i + batch_size - 1, end)]
            X, Y = X_train[:,idx], Y_train[idx]
            gradient!(cost, theta, X, Y)
            update!(rmsprop, theta)
        end
        step(progress, store_best=false)
        println(progress)
    end
    done(progress)

    println("converged after ", progress.epoch, " epochs (", round(time(progress), 2), " seconds)")
    println("testing...")
    X_test, rawY = MNIST.testdata()
    Y_test = round(Int, rawY + 1)
    n_test = length(Y_test)
    Y_pred = predict(theta, X_test)
    error = sum(Y_test .!= Y_pred)
    println("test error: $(error / n_test) ($error of $n_test)")
end

("-c" in ARGS || "--check" in ARGS) && check()
("-s" in ARGS || "--show" in ARGS) && show_example_data()
fit()

