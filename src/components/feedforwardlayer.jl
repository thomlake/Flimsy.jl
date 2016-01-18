
immutable FeedForwardLayer{T,M,N} <: Component{T,M,N}
    f::Function
    w::Variable{T,M,N}
    b::Variable{T,M,1}
end

FeedForwardLayer(f::Function, m::Int, n::Int) = FeedForwardLayer(
    f=f, 
    w=orthonormal(f, m, n), 
    b=zeros(m)
)

FeedForwardLayer(m::Int, n::Int) = FeedForwardLayer(relu, m, n)

@flimsy feedforward(theta::FeedForwardLayer, x::Variable) = theta.f(affine(theta.w, x, theta.b))

# immutable FeedForwardLayer2{T,M,N1,N2} <: Component{T,M,N1,N2}
#     f::Function
#     w::Variable{T,M,N2}
#     u::Variable{T,M,N1}
#     b::Variable{T,M,1}
# end

# FeedForwardLayer2(f::Function, m::Int, n1::Int, n2::Int) = FeedForwardLayer2(
#     f=f, 
#     w=orthonormal(f, m, n),
#     w=orthonormal(f, m, n),
#     b=zeros(m)
# )

# FeedForwardLayer(m::Int, n::Int) = FeedForwardLayer(relu, m, n)

# @flimsy feedforward(theta::FeedForwardLayer, x::Variable) = theta.f(affine(theta.w, x, theta.b))

# -- Homogenous Layer Wrapper -- #
# immutable LayerStack <: Component
#     layers::Vector{FeedForwardLayer}
# end

# function LayerStack(f::Function, n_d::Int, n_dm1::Int, n_rest::Int...)
#     dims = reverse(vcat(n_d, n_dm1, n_rest...))
#     depth = length(dims) - 1
#     layers = map(1:depth) do i
#         FeedForwardLayer(f, dims[i + 1], dims[i])
#     end
#     return LayerStack(layers)
# end

# Base.length(theta::LayerStack) = length(theta.layers)

function multilayer(::Type{FeedForwardLayer}, f::Function, n_d::Int, n_dm1::Int, n_rest::Int...)
    dims = reverse(vcat(n_d, n_dm1, n_rest...))
    depth = length(dims) - 1
    return map(1:depth) do i
        FeedForwardLayer(f, dims[i + 1], dims[i])
    end
end

@flimsy function feedforward{F<:FeedForwardLayer}(layers::Vector{F}, h)
    for theta in layers
        h = feedforward(theta, h)
    end
    return h
end
