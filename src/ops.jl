const OPERATORS = Symbol[
    :identity,
    :tanh,
    :sigmoid,
    :relu,
    :softmax,
    :wta,
    :prod,
    :linear,
    :sum,
    :minus,
    :concat,
    # :decat,
    :map,
    :affine,
    :dropout!
]

# -- Identity -- #
Base.identity(x::Var) = x

Base.identity(stack::BPStack, x::Var) = x

# -- Tanh -- #
function bwd_tanh{T,M,N}(y::Var{T,M,N}, x::Var{T,M,N})
    for i in eachindex(x)
        x.grad[i] += (1 - (y.data[i] * y.data[i])) * y.grad[i]
    end
    return nothing
end

Base.tanh(x::Var) = Var(tanh(x.data))

function Base.tanh(stack::BPStack, x::Var)
    y = tanh(x)
    push!(stack, () -> bwd_tanh(y, x))
    return y
end

# -- Sigmoid -- #
function bwd_sigmoid{T,M,N}(y::Var{T,M,N}, x::Var{T,M,N})
    for i in eachindex(x)
        x.grad[i] += y.data[i] * (1 - y.data[i]) * y.grad[i]
    end
    return nothing
end

function sigmoid(x::Var)
    y = zero(x)
    for i in eachindex(x)
        y.data[i] = 1 / (1 + exp(-x.data[i]))
    end
    return y
 end

function sigmoid(stack::BPStack, x::Var)
    y = sigmoid(x)
    push!(stack, () -> bwd_sigmoid(y, x))
    return y
end

# -- ReLU -- #
function bwd_relu{T,M,N}(y::Var{T,M,N}, x::Var{T,M,N})
    for i in eachindex(x)
        if x.data[i] > 0
            x.grad[i] += y.grad[i]
        end
    end
    return nothing
end

relu(x::Var) = Var(max(0, x.data))

function relu(stack::BPStack, x::Var)
    y = relu(x)
    push!(stack, () -> bwd_relu(y, x))
    return y
end

# -- Max -- #
# function bwd_max(y::Var, t::Number, x::Var)
#     for i = 1:endof(x.data)
#         if x.data[i] > t
#             x.grad[i] += y.grad[i]
#         end
#     end
#     return nothing
# end
#
# Base.max(t::Number, x::Var) = Var(max(t, x.data))
#
# function Base.max(stack::BPStack, t::Number, x::Var)
#     y = max(t, x)
#     push!(stack, () -> bwd_max(y, t, x))
#     return y
# end

# -- Softmax -- #
function bwd_softmax{T,M,N}(y::Var{T,M,N}, x::Var{T,M,N})
    for n = 1:size(y, 2)
        for i = 1:size(y, 1)
            for j = 1:size(y, 1)
                if i == j
                    x.grad[i,n] += y.data[i,n] * (1 - y.data[j,n]) * y.grad[j,n]
                else
                    x.grad[i,n] -= y.data[i,n] * y.data[j,n] * y.grad[j,n]
                end
            end
        end
    end
    return nothing
end

function softmax(x::Var)
    y = exp(x.data .- maximum(x.data, 1))
    y ./= sum(y, 1)
    return Var(y)
end

function softmax(stack::BPStack, x::Var)
    y = softmax(x)
    push!(stack, () -> bwd_softmax(y, x))
    return y
end

# -- Winner Takes All -- #
function bwd_wta{T,M,N}(y::Var{T,M,N}, x::Var{T,M,N})
    _, imax = findmax(x.data, 1)
    for i = 1:endof(imax)
        x.grad[imax[i]] += y.grad[imax[i]]
    end
    return nothing
end

function wta(x::Var)
    y = zero(x)
    xmax, imax = findmax(x.data, 1)
    for i = 1:endof(imax)
        y.data[imax[i]] = xmax[i]
    end
    return y
end

function wta(stack::BPStack, x::Var)
    y = wta(x)
    push!(stack, () -> bwd_wta(y, x))
    return y
end

# -- Multiply (scalar by array) -- #
function bwd_prod(y::Var, a::AbstractFloat, x::Var)
    for i in eachindex(y.grad)
        x.grad[i] += a * y.grad[i]
    end
    return nothing
end

function bwd_prod{T,M,N}(y::Var{T,M,N}, x1::Var{T,1,N}, x2::Var{T,M,N})
    for j = 1:size(x1, 2)
        for i = 1:size(y, 1)
            x1.grad[j] += y.grad[i,j] * x2.data[i,j]
        end
    end
    for j = 1:size(x2, 2)
        for i = 1:size(x2, 1)
            x2.grad[i,j] += y.grad[i,j] .* x1.data[j]
        end
    end
    return nothing
end

function bwd_prod{T}(y::Var{T}, x1::Var{T}, x2::Var{T})
    for i in eachindex(x1)
        x1.grad[i] += y.grad[i] * x2.data[i]
        x2.grad[i] += y.grad[i] * x1.data[i]
    end
    return nothing
end

Base.prod(a::AbstractFloat, x::Var) = Var(a .* x.data)

function Base.prod(stack::BPStack, a::AbstractFloat, x::Var)
    y = prod(a, x)
    push!(stack, () -> bwd_prod(y, a, x))
    return y
end

Base.prod(x1::Var, x2::Var) = Var(x1.data .* x2.data)

function Base.prod(stack::BPStack, x1::Var, x2::Var)
    y = prod(x1, x2)
    push!(stack, () -> bwd_prod(y, x1, x2))
    return y
end

# -- Linear -- #
function bwd_linear{T,M,K,N}(y::Var{T,M,N}, w::Var{T,M,K}, x::Var{T,K,N})
    dx = At_mul_B(w.data, y.grad)
    dw = A_mul_Bt(y.grad, x.data)
    for i in eachindex(x)
        x.grad[i] += dx[i]
    end
    for i in eachindex(w)
        w.grad[i] += dw[i]
    end
    # x.grad .+= At_mul_B(w.data, y.grad)
    # w.grad .+= A_mul_Bt(y.grad, x.data)
    return nothing
end

function linear{T,M,K,N}(w::Var{T,M,K}, x::Var{T,K,N})
    y = zeros(eltype(w), M, N)
    A_mul_B!(y, w.data, x.data)
    return Var(y)
end

function linear(stack::BPStack, w, x)
    y = linear(w, x)
    push!(stack, () -> bwd_linear(y, w, x))
    return y
end

# -- Sum -- #
function bwd_sum{T,M,N}(y::Var{T,M,N}, x1::Var{T,M,N}, x2::Var{T,M,N})
    for i in eachindex(y)
        x1.grad[i] += y.grad[i]
        x2.grad[i] += y.grad[i]
    end
    return nothing
end

function bwd_sum{T,M,N}(y::Var{T,M,N}, x1::Var{T,M,N}, x2::Var{T,M,1})
    for i in eachindex(x1)
        x1.grad[i] += y.grad[i]
    end
    for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            x2.grad[i] += y.grad[i,j]
        end
    end
    return nothing
end

Base.sum{T,M,N1,N2}(x1::Var{T,M,N1}, x2::Var{T,M,N2}) = Var(x1.data .+ x2.data)

function Base.sum{T,M,N1,N2}(stack::BPStack, x1::Var{T,M,N1}, x2::Var{T,M,N2})
    y = sum(x1, x2)
    push!(stack, () -> bwd_sum(y, x1, x2))
    return y
end

# -- Sum (arbitrary number of blocks) -- #
function Base.sum{T<:AbstractVar}(xs::Vector{T})
    y = sum(xs[1], xs[2])
    for i = 3:length(xs)
        y = sum(y, xs[i])
    end
    return y
end

function Base.sum{T<:AbstractVar}(stack::BPStack, xs::Vector{T})
    y = sum(stack, xs[1], xs[2])
    for i = 3:length(xs)
        y = sum(stack, y, xs[i])
    end
    return y
end

Base.sum(x1::Var, x2::Var, x3::Var, xrest::Var...) = sum([x1, x2, x3, xrest...])

Base.sum(stack::BPStack, x1::Var, x2::Var, x3::Var, xrest::Var...) = sum(stack, [x1, x2, x3, xrest...])

# -- Minus -- #
function bwd_minus{T,M,N}(y::Var{T,M,N}, a::AbstractFloat, x::Var{T,M,N})
    for i in eachindex(y)
        x.grad[i] -= y.grad[i]
    end
    return nothing
end

function bwd_minus{T,M,N}(y::Var{T,M,N}, x1::Var{T,M,N}, x2::Var{T,M,N})
    for i in eachindex(y)
        x1.grad[i] += y.grad[i]
        x2.grad[i] -= y.grad[i]
    end
    return nothing
end

minus(a::AbstractFloat, x::Var) = Var(a .- x.data)

function minus(stack::BPStack, a::AbstractFloat, x::Var)
    y = minus(a, x)
    push!(stack, () -> bwd_minus(y, a, x))
    return y
end

minus{T,M,N}(x1::Var{T,M,N}, x2::Var{T,M,N}) = Var(x1.data .- x2.data)

function minus{T,M,N}(stack::BPStack, x1::Var{T,M,N}, x2::Var{T,M,N})
    y = minus(x1, x2)
    push!(stack, () -> bwd_minus(y, x1, x2))
    return y
end

# -- Concatenation --#
function bwd_concat{T<:AbstractVar}(y::Var, xs::Vector{T})
    offset = 0
    for k = 1:length(xs)
        for j = 1:size(xs[k], 2)
            for i = 1:size(xs[k], 1)
                xs[k].grad[i,j] += y.grad[offset + i,j]
            end
        end
        offset += size(xs[k], 1)
        # j = size(inmats[b], 1)
        # inmats[b].dx .+= outmat.grad[i:i + j - 1,:]
        # i += j
    end
    return nothing
end

function concat{T<:AbstractVar}(xs::Vector{T})
    n2 = size(xs[1], 2)
    for i = 2:length(xs)
        @assert n2 == size(xs[i], 2)
    end
    return Var(vcat([x.data for x in xs]...))
end

function concat{T<:AbstractVar}(stack::BPStack, xs::Vector{T})
    y = concat(xs)
    push!(stack, () -> bwd_concat(y, xs))
    return y
end


# -- Deconcatentation -- #
# function bwd_decat(outmats::Vector{Var}, inmat::Var)
#     @assert length(outmats) == size(inmat, 1)
#     for i = 1:size(inmat, 1)
#         inmat.grad[i,:] += outmats[i].dx
#     end
#     return nothing
# end
#
# function decat(mat::Var)
#     return [Var(mat.data[i,:]) for i = 1:size(mat, 1)]
# end
#
# function decat(stack::BPStack, inmat::Var)
#     outmats = decat(inmat)
#     push!(stack, () -> bwd_decat(outmats, inmat))
#     return outmats
# end

# -- Map (single) -- #
function Base.map{T<:AbstractVar}(f::Function, xs::Vector{T})
    ys = Array(T, length(xs))
    for i = 1:length(xs)
        ys[i] = f(xs[i])
    end
    return ys
end

function Base.map{T<:AbstractVar}(stack::BPStack, f::Function, xs::Vector{T})
    ys = Array(T, length(xs))
    for i = 1:length(xs)
        ys[i] = f(stack, xs[i])
    end
    return ys
end

# -- Map (multiple) -- #
function Base.map{T<:AbstractVar}(f::Function, xss::Vector{T}...)
    n = length(xss)
    len = length(xss[1])
    for i = 2:n
        @assert len == length(xss[i])
    end
    ys = Array(T, len)
    for i = 1:len
        ys[i] = f([xss[k][i] for k = 1:n]...)
    end
    return ys
end

function Base.map{T<:AbstractVar}(stack::BPStack, f::Function, xss::Vector{T}...)
    n = length(xss)
    len = length(xss[1])
    for i = 2:n
        @assert len == length(xss[i])
    end
    ys = Array(T, len)
    for i = 1:len
        ys[i] = f(stack, [xss[k][i] for k = 1:n]...)
    end
    return ys
end

# -- Convenience Functions -- #
affine(w::Var, x::Var, b::Var) = sum(linear(w, x), b)
affine(stack::BPStack, w::Var, x::Var, b::Var) = sum(stack, linear(stack, w, x), b)

# -- Special Ops --#
function dropout!(x::Var, dp::AbstractFloat)
    data = x.data
    for i in eachindex(x)
        if rand() < dp
            data[i] = 0
        end
    end
    return x
end

function dropout!(stack::BPStack, x::Var, dp::AbstractFloat)
    data = x.data
    for i in eachindex(x)
        if rand() < dp
            data[i] = 0
        end
    end
    return x
end
