## Mixture of Gaussians
import Distributions: MixtureModel, Categorical, MultivariateNormal
import ConjugatePriors: NormalInverseWishart

immutable MixtureModelWrapper
    m::MixtureModel
end

function MoG(n_classes::Int, n_features::Int)
    d = NormalInverseWishart(zeros(n_features), 1, eye(n_features), n_features)
    mog = MixtureModel([MultivariateNormal(rand(d)...) for i = 1:n_classes], Categorical(n_classes))
    return MixtureModelWrapper(mog)
end

function Base.rand(w::MixtureModelWrapper)
    m = w.m
    y = rand(m.prior)
    x = rand(m.components[y])
    return x, y
end

function Base.rand(w::MixtureModelWrapper, n::Int)
    m = w.m
    X, Y = zeros(length(m), n), zeros(Int, n)
    for i = 1:n
        x, y = rand(w)
        X[:,i] = x
        Y[i] = y
    end
    return X, Y
end
