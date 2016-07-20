# Flimsy.jl
# Recurrent Neural Netwrks Comparison
#
# Comparison of SRNN, LSTM, and GRU hidden layers
# on sequential xor task. Demonstrates how to build
# components with a generic sub-component.
using Synthetic
using Flimsy
using Flimsy.Components
import Flimsy.Components: cost, predict

"""
Sequence Tagger Params

    h[t] = g(x[t], h[t-1]) : Recurrent Hidden Layer
    y[t] = f(h[t])         : Classifier
"""
immutable Params{R<:RecurrentComponent} <: Component
    clf::SoftmaxRegression
    rnn::R
end
# Params{V}(clf::SoftmaxRegression{V}, rnn::RecurrentComponent{V}) = Params{V}(clf, rnn)

predict(scope::Scope, params::Params, xs::Vector) = @with scope [predict(params.clf, h) for h in unfold(params.rnn, xs)]

function cost(scope::Scope, params::Params, xs::Vector, ys::Vector)
    nll = 0.0
    @with scope begin
        for (h, y) in zip(unfold(params.rnn, xs), ys)
            nll += cost(params.clf, h, y)
        end
    end
    return nll
end

Params{R<:RecurrentComponent}(::Type{R}, n_out::Int, n_hid::Int, n_in::Int) = Params(
    clf=SoftmaxRegression(n_out, n_hid),
    rnn=R(n_hid, n_out),
)

const recurrent_layer_types = [SimpleRecurrent, GatedRecurrent, Lstm]

function check()
    n_out, n_hid, n_in = 2, 5, 2
    x, y = rand(Synthetic.XORTask(20))
    for R in recurrent_layer_types
        params = Params(R, n_out, n_hid, n_in)
        println(R.name)
        println(params)
        check_gradients(cost, params, x, y)
    end
end

function count_errors(y_pred, y_true)
    count = 0
    length(y_true) == length(y_pred) || error("sequence length mismatch")
    for t = 1:length(y_true)
        if y_true[t] != y_pred[t][1]
            count += 1
        end
    end
    return count
end

function main()
    srand(1235)
    n_out, n_hid, n_in = 2, 5, 2

    # Build train and valid datasets
    n_train, n_valid = 50, 20
    xor_train = Synthetic.XORTask(5:20)
    data_train = rand(xor_train, n_train)
    data_valid = rand(xor_train, n_valid)

    n_timesteps_train, min_len_train, max_len_train = 0, typemax(Int), typemin(Int)
    for (x, y) in data_train
        n_timesteps_train += length(y)
        min_len_train = min(min_len_train, length(y))
        max_len_train = max(max_len_train, length(y))
    end

    println("[train data]")
    println("  number samples   => ", n_train)
    println("  number timesteps => ", n_timesteps_train)
    println("  min seq length   => ", min_len_train)
    println("  max seq length   => ", max_len_train)

    # Build test dataset
    n_test = 20
    xor_test = Synthetic.XORTask(100)
    data_test = rand(xor_test, n_test)
    n_timesteps_test, min_len_test, max_len_test = 0, typemax(Int), typemin(Int)
    for (x, y) in data_test
        n_timesteps_test += length(y)
        min_len_test = min(min_len_test, length(y))
        max_len_test = max(max_len_test, length(y))
    end

    println("[test data]")
    println("  number samples   => ", n_test)
    println("  number timesteps => ", n_timesteps_test)
    println("  min seq length   => ", min_len_test)
    println("  max seq length   => ", max_len_test)

    for R in recurrent_layer_types
        params = Params(R, n_out, n_hid, n_in)
        opt = optimizer(GradientDescent, params, learning_rate=0.1, clip=1.0, clipping_type=:scale)
        n_params = sum(map(x -> prod(size(x)), convert(Vector, params)))
        println("[", typeof(params.rnn).name, "]")
        indices = collect(1:n_train)
        max_epochs, n_epochs = 200, 0
        start_time = time()
        while n_epochs < max_epochs
            n_epochs += 1
            shuffle!(indices)
            nll = 0.0
            for i in indices
                x, y = data_train[i]
                nll += @backprop cost(params, x, y)
                update!(opt)
            end
            errors = 0
            for (x, y_true) in data_valid
                y_pred = @run predict(params, x)
                errors += count_errors(y_pred, y_true)
            end
            errors == 0 && break
        end

        stop_time = time()
        errors = 0
        for (x, y_true) in data_test
            y_pred = @run predict(params, x)
            errors += count_errors(y_pred, y_true)
        end

        println("  wall time              => ", stop_time - start_time)
        println("  number of epochs       => ", n_epochs)
        println("  number of parameters   => ", n_params)
        println("  total errors           => ", errors)
        println("  avg error per sequence => ", errors / n_test)
        println("  avg error per timestep => ", errors / n_timesteps_test)
    end
end

("-c" in ARGS || "--check" in ARGS) && check()
main()
