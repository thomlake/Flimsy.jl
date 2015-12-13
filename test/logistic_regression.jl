using Flimsy
using Flimsy: Components
using Base.Test

function check()
    n_classes, n_features, n_samples = 3, 20, 10
    X, Y = rand(Flimsy.Demo.MoG(n_classes, n_features), n_samples)
    theta = LogisticRegression(n_classes, n_features)
    g() = gradient!(cost, theta, X, Y)
    c() = cost(theta, X, Y)
    gradcheck(g, c, theta, verbose=false)
end

function fit()
    n_classes, n_features = 3, 20
    n_train, n_test = 50, 50
    D = Flimsy.Demo.MoG(n_classes, n_features)
    X_train, Y_train = rand(D, n_train)
    X_test, Y_test = rand(D, n_test)

    theta = LogisticRegression(n_classes, n_features)
    rmsprop = optimizer(RMSProp, theta, learning_rate=0.01, decay=0.9)
    nll_prev = Inf
    for i = 1:50
        nll_curr = gradient!(cost, theta, X_train, Y_train)
        update!(rmsprop, theta)
        @test nll_curr < nll_prev
        nll_prev = nll_curr
    end
    @test sum(Y_test .!= predict(theta, X_test)) / n_test < 0.4
end

check()
fit()
