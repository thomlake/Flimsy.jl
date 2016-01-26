## Mixture of Gaussians
import Distributions: MixtureModel, Categorical, MultivariateNormal
import ConjugatePriors: NormalInverseWishart

immutable MoG
    model::MixtureModel
end

function MoG(n_classes::Int, n_features::Int)
    d = NormalInverseWishart(zeros(n_features), 1, eye(n_features), 2 * n_features)
    model = MixtureModel([MultivariateNormal(rand(d)...) for i = 1:n_classes], Categorical(n_classes))
    return MoG(model)
end

function Base.rand(mog::MoG)
    y = rand(mog.model.prior)
    x = rand(mog.model.components[y])
    return x, y
end

function Base.rand(mog::MoG, n::Int)
    X, Y = zeros(length(mog.model), n), zeros(Int, n)
    for i = 1:n
        x, y = rand(mog)
        X[:,i] = x
        Y[i] = y
    end
    return X, Y
end
