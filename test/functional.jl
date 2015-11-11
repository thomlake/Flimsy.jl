using Nimble

function main()
    println("[linreg]")
    n_output, n_features = 2, 3
    w = randn(n_output, n_features)
    b = randn(n_output)
    clf = Learnable(components.LinearRegression(w, b))

    x = randn(n_features)
    y = randn(n_output)
    p = w * x .+ b
    println(p)
    println(clf.theta(x))
    println(clf(x, y))
    println(p .- y)
    backprop!(clf)
    println()

    println("[logreg]")
    n_output, n_features = 2, 3
    w = randn(n_output, n_features)
    b = randn(n_output)
    clf = Learnable(components.LogisticRegression(w, b))

    x = randn(n_features)
    y = rand(1:n_output)
    p = softmax(NimMat(w * x .+ b)).x
    println(p)
    println(clf.theta(x))
    println(clf(x, y))
    p[y] -= 1
    println(p)
    backprop!(clf)

    for param in getparams(clf)
        println(param)
    end

    for (name, param) in getnamedparams(clf)
        println((name, param))
    end

end

main()
