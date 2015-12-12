# Flimsy.jl
# Attention Demo

# Task:
# Given a vector of (number, flag) pairs 
# learn to output the sum of the two 
# numbers where the flag is non-zero.

# Example:
# Input = [[2,1],[5,0],[9,0],[4,0],[4,0],[4,0],[7,0],[9,0],[4,1]]
# Output = 6

using Flimsy
using Flimsy.Components
import Flimsy.Components: feedforward, predict, cost
import Flimsy.Demo: AddTask

immutable Params{T,NHid} <: Component
    output::LinearRegression{T,1,NHid}
    attend::FeedForwardLayer{T,1,NHid}
    hidden::FeedForwardLayer{T,NHid,2}
end

function Params(n_hid::Int)
    output = LinearRegression(1, n_hid)
    attend = FeedForwardLayer(identity, Glorot(1, n_hid), Zeros(1))
    hidden = FeedForwardLayer(relu, Glorot(n_hid, 2), Zeros(n_hid))
    return Params(output, attend, hidden)
end

@flimsy function feedforward(theta::Params, xs::Vector)
    hs = Variable[feedforward(theta.hidden, x) for x in xs]
    as = softmax(Variable[feedforward(theta.attend, h) for h in hs])
    return sum(Variable[prod(a, h) for (a, h) in zip(as, hs)])
end

@flimsy predict(theta::Params, xs::Vector) = predict(theta.output, feedforward(theta, xs))

@flimsy cost(theta::Params, xs::Vector, y::Number) = cost(theta.output, feedforward(theta, xs), y)

function check()
    n_out, n_hid, n_in = 1, 7, 2
    x, y = rand(AddTask(5:5))
    theta = Params(n_hid)
    g() = gradient!(cost, theta, x, y)
    c() = cost(theta, x, y)
    gradcheck(g, c, theta, tol=1e-3)
end

function fit()
    srand(123)
    n_train = 100
    n_valid = 20
    n_hid = 20
    addtask = AddTask(5:20)

    X_train, Y_train = rand(addtask, n_train)
    X_valid, Y_valid = rand(addtask, n_valid)
    minlen = minimum(map(length, X_train))
    maxlen = maximum(map(length, X_train))

    theta = Params(n_hid)
    opt = optimizer(RMSProp, theta, learning_rate=0.01, decay=0.95, clip=5.0, clipping_type=:scale)
    progress = Progress(theta, max_epochs=50, patience=10) do
        mse = 0.0
        for (xs, y) in zip(X_valid, Y_valid)
            mse += cost(theta, xs, y)
        end
        return mse / n_valid
    end

    indices = collect(1:n_train)
    println("fitting...")
    start(progress)
    while !quit(progress)
        shuffle!(indices)
        for i in indices
            gradient!(cost, theta, X_train[i], Y_train[i])
            update!(opt, theta)
        end
        step(progress, store_best=true)
        if progress.frustration > 4
            opt.learning_rate *= 0.5
        end
        println(progress)
    end
    done(progress)

    println("[train]")
    println("  number of samples => ", n_train)
    println("  number of epochs  => ", progress.epoch)
    println("  cpu time          => ", round(time(progress), 2), " seconds")
    println("  min seq length    => ", minlen)
    println("  max seq length    => ", maxlen)
    println("  mean square error => ", progress.current_value)

    n_test = 20
    X_test, Y_test = rand(AddTask(20:50), n_test)
    minlen = minimum(map(length, X_test))
    maxlen = maximum(map(length, X_test))
    theta_best = progress.best_model
    mse = 0
    for (xs, y) in zip(X_test, Y_test)
        p = predict(theta_best, xs)
        p.data[1] = round(p.data[1])
        mse += Flimsy.Cost.gauss(y, p)
    end
    println("[test -- best validation]")
    println("  number of samples => ", n_test)
    println("  min seq length    => ", minlen)
    println("  max seq length    => ", maxlen)
    println("  square error      => ", mse)
    println("  mean square error => ", mse / n_test)
end

check()
fit()
