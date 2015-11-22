using Flimsy
using Flimsy: Components
using Base.Test

function check()
    n_classes, n_features, n_samples = 3, 20, 10
    X, Y = rand(Flimsy.SampleData.MoG(n_classes, n_features), n_samples)
    theta = LogisticRegression(n_classes, n_features)
    g() = gradient!(probs, theta, X, Y)
    c() = probs(theta, X, Y)[1]
    gradcheck(g, c, theta, verbose=false)
end

function fit()
    n_classes, n_features = 3, 20
    n_train, n_test = 50, 50
    D = Flimsy.SampleData.MoG(n_classes, n_features)
    X_train, Y_train = rand(D, n_train)
    X_test, Y_test = rand(D, n_test)

    theta = LogisticRegression(n_classes, n_features)
    rmsprop = optimizer(RMSProp, theta, learning_rate=0.01, decay=0.9)
    nll_prev = Inf
    for i = 1:50
        nll_curr = gradient!(probs, theta, X_train, Y_train)[1]
        update!(rmsprop, theta)
        @test nll_curr < nll_prev
        nll_prev = nll_curr
    end
    @test sum(Y_test .!= predict(theta, X_test)) / n_test < 0.4
end

check()
fit()
