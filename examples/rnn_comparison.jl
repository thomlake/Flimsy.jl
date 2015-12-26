# Flimsy/examples/rnn_comparisson.jl
#
# Comparison of SRNN, LSTM, and GRU hidden layers
# on sequential xor task. Demonstrates how to build
# components with generic sub-components.

using Flimsy
using Flimsy.Components
import Flimsy.Components: cost, predict
import Flimsy.Demo: XORTask

# Sequence tagger:
# h[t] = g(x[t], h[t-1]) : Recurrent Hidden Layer
# y[t] = f(h[t])         : Classifier
immutable Params{T,NOut,NHid,NIn} <: Component
    out::LogisticRegression{T,NOut,NHid}
    rnn::RecurrentComponent{T,NHid,NIn}
end

@flimsy predict(theta::Params, xs::Vector) = [predict(theta.out, h)[1] for h in unfold(theta.rnn, xs)]

@flimsy function cost(theta::Params, xs::Vector, ys::Vector)
    nll = 0.0
    for (h, y) in zip(unfold(theta.rnn, xs), ys)
        nll += cost(theta.out, h, y)
    end
    return nll
end

Params{R<:RecurrentComponent}(::Type{R}, n_out::Int, n_hid::Int, n_in::Int) = Params(
    LogisticRegression(n_out, n_hid),
    R(n_hid, n_in),
)

function check()
    n_out, n_hid, n_in = 2, 5, 2
    xs, ys = rand(XORTask(20))
    for R in (SimpleRecurrent, GatedRecurrent, LSTM)
        print("$R => ")
        theta = Params(R, n_out, n_hid, n_in)
        g() = gradient!(cost, theta, xs, ys)
        c() = cost(theta, xs, ys)
        gradcheck(g, c, theta)
    end
end

function fit()
    srand(123)

    n_out, n_hid, n_in = 2, 5, 2
    n_train, n_valid = 100, 20
    xor = XORTask(5:20)
    X_train, Y_train = rand(xor, n_train)
    X_valid, Y_valid = rand(xor, n_valid)
    n_train_timesteps = mapreduce(length, +, Y_train)
    minlen = minimum(map(length, X_train))
    maxlen = maximum(map(length, X_train))

    println("[train]")
    println("  number samples   => ", n_train)
    println("  number timesteps => ", n_train_timesteps)
    println("  min seq length   => ", minlen)
    println("  max seq length   => ", maxlen)

    models = [R => Params(R, n_out, n_hid, n_in) for R in (SimpleRecurrent, GatedRecurrent, LSTM)]
    info = Dict()

    for (name, theta) in models
        println(name)
        opt = optimizer(GradientDescent, theta, learning_rate=0.1, clip=3.0, clipping_type=:scale)
        progress = Progress(theta, min_epochs=100, max_epochs=100) do
            errors = 0
            for (xs, ys) in zip(X_valid, Y_valid)
                errors += sum(ys .!= predict(theta, xs))
            end
            return errors
        end

        indices = collect(1:n_train)
        start(progress)
        while !quit(progress)
            shuffle!(indices)
            for i in indices
                xs, ys = X_train[i], Y_train[i]
                gradient!(cost, theta, xs, ys)
                update!(opt, theta)
            end
            step(progress, store_best=true)
            progress.epoch % 20 == 0 && println(progress)
        end
        done(progress)
        info[name] = progress
        println()
    end

    n_test = 20
    X_test, Y_test = rand(XORTask(100:200), n_test)
    n_test_timesteps = mapreduce(length, +, Y_test)
    minlen = minimum(map(length, X_test))
    maxlen = maximum(map(length, X_test))


    println("[test]")
    println("  number samples   => ", n_test)
    println("  number timesteps => ", n_test_timesteps)
    println("  min seq length   => ", minlen)
    println("  max seq length   => ", maxlen)
    for (name, progress) in info
        theta = progress.best_model
        errors = 0
        for (xs, ys) in zip(X_test, Y_test)
            errors += sum(ys .!= predict(theta, xs))
        end
        n_params = sum([prod(size(p)) for p in getparams(theta)])
        println(name)
        println("  params    => ", n_params)
        println("  cpu time  => ", time(progress))
        println("  sum error => ", errors)
        println("  seq error => ", errors / n_test)
        println("  avg error => ", errors / n_test_timesteps)
    end
end

check()
fit()
