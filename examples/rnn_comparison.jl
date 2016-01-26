# Flimsy.jl
# RNN Comparison
#
# Comparison of SRNN, LSTM, and GRU hidden layers
# on sequential xor task. Demonstrates how to build
# components with a generic sub-component.

using Flimsy
using Flimsy.Components
import Flimsy.Components: cost, predict
import Flimsy.Demo: XORTask

# Sequence Tagger:
# h[t] = g(x[t], h[t-1]) : Recurrent Hidden Layer
# y[t] = f(h[t])         : Classifier
immutable Params{V<:Variable} <: Component{V}
    clf::SoftmaxRegression{V}
    rnn::RecurrentComponent{V}
end

@component predict(params::Params, xs::Vector) = [predict(params.clf, h)[1] for h in unfold(params.rnn, xs)]

@component function cost{I<:Integer}(params::Params, xs::Vector, ys::Vector{I})
    nll = 0.0
    for (h, y) in zip(unfold(params.rnn, xs), ys)
        nll += cost(params.clf, h, y)
    end
    return nll
end

Params{R<:RecurrentComponent}(::Type{R}, n_out::Int, n_hid::Int, n_in::Int) = Params(
    clf=SoftmaxRegression(n_out, n_hid),
    rnn=R(n_hid, n_out),
)

const recurrent_layer_types = (SimpleRecurrent, GatedRecurrent, LSTM)#, ResidualRecurrent)

function check()
    n_out, n_hid, n_in = 2, 5, 2
    xs, ys = rand(XORTask(20))
    for R in recurrent_layer_types
        println("$R")
        println("  params [")
        params = Params(R, n_out, n_hid, n_in)
        param_count = 0
        for (name, param) in getnamedparams(params)
            param_count += prod(size(param))
            println("    $name => $(size(param)),")
        end
        println("  ]")
        println("  count: $param_count")
        print("  ")
        g = () -> gradient!(cost, params, map(Input, xs), ys)
        c = () -> cost(params, map(Input, xs), ys)
        check_gradients(g, c, params)
    end
end

function fit()
    srand(1235)

    n_out, n_hid, n_in = 2, 5, 2
    n_train, n_valid = 50, 20
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

    models = [R => Params(R, n_out, n_hid, n_in) for R in recurrent_layer_types]
    info = Dict()

    for (name, params) in models
        println(name)
        indices = collect(1:n_train)
        opt = optimizer(GradientDescent, params, learning_rate=0.1, clip=1.0, clipping_type=:scale)
        evaluate = FunctionEvaluation() do
            errors = 0
            for (xs, y_true) in zip(X_valid, Y_valid)
                y_pred = predict(params, map(Input, xs))
                for t = 1:length(y_true)
                    errors += y_pred[t] == y_true[t][1] ? 0 : 1
                end
            end
            return errors
        end
        progress = Progress(params, evaluate, Patience(Inf), min_epochs=30, max_epochs=30, frequency=5)
        while !converged(progress)
            shuffle!(indices)
            for i in indices
                xs, ys = X_train[i], Y_train[i]
                gradient!(cost, params, map(Input, xs), ys)
                update!(opt, params)
            end
            progress(save=true) && println(progress)
        end
        timer_stop(progress)
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
        params = progress.best_model
        errors = 0
        for (xs, y_true) in zip(X_valid, Y_valid)
            y_pred = predict(params, map(Input, xs))
            for t = 1:length(y_true)
                errors += y_pred[t] == y_true[t][1] ? 0 : 1
            end
        end
        param_count = sum([prod(size(p)) for p in getparams(params)])
        println(name)
        println("  params    => ", param_count)
        println("  cpu time  => ", time(progress))
        println("  sum error => ", errors)
        println("  seq error => ", errors / n_test)
        println("  avg error => ", errors / n_test_timesteps)
    end
end

("-c" in ARGS || "--check" in ARGS) && check()
fit()
