# Flimsy.jl
# Recurrent Neural Netwrks Comparison
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

@component predict(params::Params, xs::Vector) = [predict(params.clf, h) for h in unfold(params.rnn, xs)]

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

const recurrent_layer_types = [SimpleRecurrent, GatedRecurrent, LSTM]

function check()
    n_out, n_hid, n_in = 2, 5, 2
    xs, ys = rand(XORTask(20))
    scope = Scope()
    for R in recurrent_layer_types
        println(R.name)
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
        g = () -> gradient!(cost, reset!(scope), params, map(Input, xs), ys)
        c = () -> cost(reset!(scope), params, map(Input, xs), ys)
        check_gradients(g, c, params)
    end
end

function error_count(scope, params, X, Y)
    errors = 0
    for (xs, y_true) in zip(X, Y)
        y_pred = predict(reset!(scope), params, map(Input, xs))
        for t = 1:length(y_true)
            errors += y_true[t] == y_pred[t][1] ? 0 : 1
        end
    end
    return errors
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
    scope = Scope()
    for (R, params) in models
        println(R.name)
        indices = collect(1:n_train)
        opt = optimizer(GradientDescent, params, learning_rate=0.1, clip=1.0, clipping_type=:scale)
        evaluate = FunctionEvaluation(() -> error_count(scope, params, X_valid, Y_valid))
        progress = Progress(params, evaluate, Patience(Inf), min_epochs=30, max_epochs=30, frequency=5)
        
        while !converged(progress)
            shuffle!(indices)
            for i in indices
                xs, ys = X_train[i], Y_train[i]
                gradient!(cost, reset!(scope), params, map(Input, xs), ys)
                update!(opt)
            end
            progress(save=true) && println(progress)
        end
        timer_stop(progress)
        
        info[R] = progress
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
    for (R, progress) in info
        params = progress.best_model
        errors = error_count(scope, params, X_test, Y_test)
        param_count = sum([prod(size(p)) for p in getparams(params)])
        println(R.name)
        println("  params    => ", param_count)
        println("  cpu time  => ", time(progress))
        println("  sum error => ", errors)
        println("  seq error => ", errors / n_test)
        println("  avg error => ", errors / n_test_timesteps)
    end
end

("-c" in ARGS || "--check" in ARGS) && check()
fit()
