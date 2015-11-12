using Flimsy
using Flimsy.Components
import Flimsy.Components: score, predict, probs
import MNIST

immutable NNet <: Component
    output::LogisticRegression
    hidden::LayerStack
end

@flimsy score(nnet::NNet, x) = score(nnet.output, feedforward(nnet.hidden, x))

@flimsy predict(nnet::NNet, x) = predict(nnet.output, feedforward(nnet.hidden, x))

@flimsy probs(nnet::NNet, x) = probs(nnet.output, feedforward(nnet.hidden, x))

@flimsy probs(nnet::NNet, x, y) = probs(nnet.output, feedforward(nnet.hidden, x), y)

function image_string(x::Vector, symbols::Vector{Char}=['-', '+'])
    s = isqrt(length(x))
    img = reshape(x, (s, s))
    rows = ASCIIString[]
    for i in 1:s
        push!(rows, join(symbols[int(img[i,:] .> 0) + 1]))
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
    hidden = LayerStack(relu, 5, 10, 20, size(X, 1))
    output = LogisticRegression(10, 5)
    theta = NNet(output, hidden)
    g() = gradient!(probs, theta, X, Y)
    c() = probs(theta, X, Y)[1]
    gradcheck(g, c, theta)
end

function show_example_data()
    rawX, rawY = MNIST.traindata()
    X = extras.zscore(rawX)
    @assert all(isfinite(X))
    for i = 1:10
        println(int(rawY[i]))
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

    hidden = LayerStack(relu, 50, 100, 100, size(X_train, 1))
    output = LogisticRegression(10, 50)
    theta = NNet(output, hidden)

    rmsprop = optimizer(RMSProp, theta, learning_rate=0.01 / batch_size, decay=0.8)

    progress = Flimsy.Extras.Progress(theta, patience=10) do
        return sum(Y_valid .!= predict(theta, X_valid)) / n_valid
    end

    function statusmsg()
        @printf("epoch: %03d, frustration: %02d, best: %0.02f, curr: %0.02f\n", progress.epoch, progress.frustration, progress.best_value, progress.current_value)
    end

    println("fitting...")
    start(progress)
    while !quit(progress)
        shuffle!(indices)
        for i = 1:batch_size:n_train
            idx = indices[i:min(i + batch_size - 1, end)]
            X, Y = X_train[:,idx], Y_train[idx]
            gradient!(probs, theta, X, Y)
            update!(rmsprop, theta)
        end
        step(progress, progress.epoch % 10 == 0)
        statusmsg()
    end
    done(progress)

    println("converged after ", progress.epoch, " epochs (", round(time(progress), 2), " seconds)")
    println("testing...")
    X_test, rawY = MNIST.testdata()
    # teX = nnx.zscore(rawX, mu, sigma)
    Y_test = round(Int, rawY + 1)
    n_test = length(Y_test)
    Y_pred = predict(theta, X_test)
    error = sum(Y_test .!= Y_pred)
    println("test error: $(error / n_test) ($error of $n_test)")
end

check()
fit()
# show_example_data()
