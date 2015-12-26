
immutable FeedForwardLayer{T,M,N} <: Component{T,M,N}
    f::Function
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

FeedForwardLayer(f::Function, w::Matrix, b::Vector) = FeedForwardLayer(f, Flimsy.Variable(w), Flimsy.Variable(b))

FeedForwardLayer(f::Function, m::Int, n::Int) = FeedForwardLayer(f, Orthonormal(m, n), Zeros(m))

FeedForwardLayer(w::Matrix, b::Vector) = FeedForwardLayer(relu, w, b)

FeedForwardLayer(m::Int, n::Int) = FeedForwardLayer(relu, Orthonormal(sqrt(2), m, n), Zeros(m))

@flimsy feedforward(theta::FeedForwardLayer, x::Variable) = theta.f(affine(theta.w, x, theta.b))

# -- Homogenous Layer Wrapper -- #
immutable LayerStack <: Component
    layers::Vector{FeedForwardLayer}
end

function LayerStack(f::Function, n_d::Int, n_dm1::Int, n_rest::Int...)
    dims = reverse(vcat(n_d, n_dm1, n_rest...))
    depth = length(dims) - 1
    layers = map(1:depth) do i
        FeedForwardLayer(f, dims[i + 1], dims[i])
    end
    return LayerStack(layers)
end

Base.length(theta::LayerStack) = length(theta.layers)

@flimsy function feedforward(theta::LayerStack, h)
    for layer in theta.layers
        h = feedforward(layer, h)
    end
    return h
end
