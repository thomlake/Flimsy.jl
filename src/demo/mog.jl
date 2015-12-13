## Mixture of Gaussians
import Distributions: Categorical, MultivariateNormal, InverseWishart

type MoG
    p::Categorical
    d::Vector{MultivariateNormal}
end

function MoG{T<:Float64}(mu::Vector{Vector{T}}, sigma::Vector{Matrix{T}})
    n = length(mu)
    return MoG(Categorical(n), [MultivariateNormal(mu[i], sigma[i]) for i = 1:n])
end

function MoG(n_classes::Int, n_features::Int)
    mu = [randn(n_features) for i = 1:n_classes]
    sigma = [rand(InverseWishart(n_features + 1, eye(n_features))) for i = 1:n_classes]
    return MoG(mu, sigma)
end

Base.length(mog::MoG) = length(mog.d[1])

function Base.rand(mog::MoG)
    y = rand(mog.p)
    x = rand(mog.d[y])
    return x, y
end

function Base.rand(mog::MoG, n::Int)
    X, Y = zeros(length(mog), n), zeros(Int, n)
    for i = 1:n
        x, y = rand(mog)
        X[:,i] = x
        Y[i] = y
    end
    return X, Y
end
