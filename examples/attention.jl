using Flimsy
using Flimsy.Components
import Flimsy.Components: feedforward, predict, cost

immutable AddTask
    range::UnitRange{Int}
end

AddTask(t::Int) = AddTask(t:t)

function Base.rand(addtask::AddTask)
    steps = rand(addtask.range)
    i1 = rand(1:steps)
    i2 = i1
    while i1 == i2
        i2 = rand(1:steps)
    end
    n1 = rand(1:10)
    n2 = rand(1:10)
    output = float(n1 + n2)
    input = Vector{Float64}[]

    for i = 1:steps
        x = if i == i1
            [n1, 1]
        elseif i == i2
            [n2, 1]
        else
            [rand(1:10), 0]
        end
        push!(input, x)
    end

    return input, output
end

function Base.rand(addtask::AddTask, n::Int)
    x, y = rand(addtask)
    X, Y = typeof(x)[x], typeof(y)[y]
    for i = 2:n
        x, y = rand(addtask)
        push!(X, x)
        push!(Y, y)
    end
    X, Y
end

immutable DifferentiableFilter{T,NHid} <: Component
    output::LinearRegression{T,1,NHid}
    scorer::FeedForwardLayer{T,1,NHid}
    hidden::FeedForwardLayer{T,NHid,2}
end

function DifferentiableFilter(n_hid::Int)
    output = LinearRegression(1, n_hid)
    scorer = FeedForwardLayer(identity, Orthonormal(1.0, 1, n_hid), Zeros(1))
    hidden = FeedForwardLayer(relu, Orthonormal(sqrt(2), n_hid, 2), Zeros(n_hid))
    return DifferentiableFilter(output, scorer, hidden)
end

@flimsy function feedforward(nnet::DifferentiableFilter, xs::Vector)
    # traditional attention
    hs = Variable[feedforward(nnet.hidden, xs[t]) for t = 1:length(xs)]
    as = softmax(Variable{Array{Float64,2},1,1}[feedforward(nnet.scorer, hs[t]) for t = 1:length(hs)])
    return sum(Variable[prod(a, h) for (a, h) in zip(as, hs)])
    
    # additive attention
    # hs = [feedforward(nnet.hidden, xs[t]) for t = 1:length(xs)]
    # as = [feedforward(nnet.scorer, hs[t]) for t = 1:length(hs)]
    # return sum(Variable[prod(a, h) for (a, h) in zip(as, hs)])
end

@flimsy predict(nnet::DifferentiableFilter, xs::Vector) = predict(nnet.output, feedforward(nnet, xs))

@flimsy cost(nnet::DifferentiableFilter, xs::Vector, y::Number) = cost(nnet.output, feedforward(nnet, xs), y)

function check()
    n_in = 2
    n_out = 1
    x, y = rand(AddTask(5:5))
    theta = DifferentiableFilter(7)
    g() = gradient!(cost, theta, x, y)
    c() = cost(theta, x, y)[1]
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

    theta = DifferentiableFilter(n_hid)

    opt = optimizer(RMSProp, theta, learning_rate=0.01, decay=0.95, clip=5.0, clipping_type=:scale)

    progress = Flimsy.Extras.Progress(theta, max_epochs=500, patience=10) do
        mse = 0.0
        for (xs, y) in zip(X_valid, Y_valid)
            mse += cost(theta, xs, y)
        end
        return mse / n_valid
    end

    function statusmsg()
        @printf("epoch: %03d, frustration: %02d, best: %0.02f, curr: %0.02f\n", progress.epoch, progress.frustration, progress.best_value, progress.current_value)
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
        step(progress, true)
        if progress.frustration > 4
            opt.learning_rate *= 0.5
        end
        statusmsg()
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
