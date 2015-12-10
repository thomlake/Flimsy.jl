# -- Identity -- #
Base.identity(x::Variable) = x

Base.identity(stack::BPStack, x::Variable) = x

# -- Tanh -- #
function bwd_tanh{T,M,N}(y::Variable{T,M,N}, x::Variable{T,M,N})
    for i in eachindex(x)
        x.grad[i] += (1 - (y.data[i] * y.data[i])) * y.grad[i]
    end
    return nothing
end

Base.tanh(x::Variable) = Variable(tanh(x.data))

function Base.tanh(stack::BPStack, x::Variable)
    y = tanh(x)
    push!(stack, () -> bwd_tanh(y, x))
    return y
end

# -- Sigmoid -- #
function bwd_sigmoid{T,M,N}(y::Variable{T,M,N}, x::Variable{T,M,N})
    for i in eachindex(x)
        x.grad[i] += y.data[i] * (1 - y.data[i]) * y.grad[i]
    end
    return nothing
end

# x * (1 - x) * g = (i - x)
# x * (1 - x) / (i - x) = 1 / g
# (i - x) / x * (1 - x) = g

function sigmoid(x::Variable)
    y = zero(x)
    for i in eachindex(x)
        y.data[i] = 1 / (1 + exp(-x.data[i]))
    end
    return y
 end

function sigmoid(stack::BPStack, x::Variable)
    y = sigmoid(x)
    push!(stack, () -> bwd_sigmoid(y, x))
    return y
end

# -- ReLU -- #
function bwd_relu{T,M,N}(y::Variable{T,M,N}, x::Variable{T,M,N})
    for i in eachindex(x)
        if x.data[i] > 0
            x.grad[i] += y.grad[i]
        end
    end
    return nothing
end

relu(x::Variable) = Variable(max(0, x.data))

function relu(stack::BPStack, x::Variable)
    y = relu(x)
    push!(stack, () -> bwd_relu(y, x))
    return y
end

# -- Max -- #
# function bwd_max(y::Variable, t::Number, x::Variable)
#     for i = 1:endof(x.data)
#         if x.data[i] > t
#             x.grad[i] += y.grad[i]
#         end
#     end
#     return nothing
# end
#
# Base.max(t::Number, x::Variable) = Variable(max(t, x.data))
#
# function Base.max(stack::BPStack, t::Number, x::Variable)
#     y = max(t, x)
#     push!(stack, () -> bwd_max(y, t, x))
#     return y
# end

# -- Softmax -- #
function bwd_softmax{T,M,N}(y::Variable{T,M,N}, x::Variable{T,M,N})
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

function softmax(x::Variable)
    y = exp(x.data .- maximum(x.data, 1))
    y ./= sum(y, 1)
    return Variable(y)
end

function softmax(stack::BPStack, x::Variable)
    y = softmax(x)
    push!(stack, () -> bwd_softmax(y, x))
    return y
end

# -- Softmax over vector of 1x1 -- #
function bwd_softmax{V<:Variable}(ys::Vector{V}, xs::Vector{V})
    n = length(ys)
    for i = 1:n
        for j = 1:n
            if i == j
                xs[i].grad[1] += ys[i].data[1] * (1 - ys[j].data[1]) * ys[j].grad[1]
            else
                xs[i].grad[1] -= ys[i].data[1] * ys[j].data[1] * ys[j].grad[1]
            end
        end
    end
    return nothing
end

function softmax{V<:Variable}(xs::Vector{V})
    all(x -> size(x) == (1, 1), xs) || error("softmax can only be applied over vectors if elements are of size (1, 1)")
    xmax = -Inf
    for x in xs
        if x.data[1] > xmax
            xmax = x.data[1]
        end
    end

    Z = 0.0
    ys = Variable{typeof(xs[1].data),1,1}[zero(x) for x in xs]
    for i in eachindex(xs)
        e_xi = exp(xs[i].data[1] - xmax)
        ys[i].data[1] = e_xi
        Z += e_xi
    end

    for i in eachindex(ys)
        ys[i].data[1] /= Z
    end

    return ys
end

function softmax{V<:Variable}(stack::BPStack, xs::Vector{V})
    ys = softmax(xs)
    push!(stack, () -> bwd_softmax(ys, xs))
    return ys
end

# -- Winner Takes All -- #
function bwd_wta{T,M,N}(y::Variable{T,M,N}, x::Variable{T,M,N})
    _, imax = findmax(x.data, 1)
    for i = 1:endof(imax)
        x.grad[imax[i]] += y.grad[imax[i]]
    end
    return nothing
end

function wta(x::Variable)
    y = zero(x)
    xmax, imax = findmax(x.data, 1)
    for i = 1:endof(imax)
        y.data[imax[i]] = xmax[i]
    end
    return y
end

function wta(stack::BPStack, x::Variable)
    y = wta(x)
    push!(stack, () -> bwd_wta(y, x))
    return y
end

# -- Multiply (scalar by array) -- #
function bwd_prod(y::Variable, a::AbstractFloat, x::Variable)
    for i in eachindex(y.grad)
        x.grad[i] += a * y.grad[i]
    end
    return nothing
end

function bwd_prod{T,M,N}(y::Variable{T,M,N}, x1::Variable{T,1,N}, x2::Variable{T,M,N})
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

function bwd_prod{T}(y::Variable{T}, x1::Variable{T}, x2::Variable{T})
    for i in eachindex(x1)
        x1.grad[i] += y.grad[i] * x2.data[i]
        x2.grad[i] += y.grad[i] * x1.data[i]
    end
    return nothing
end

Base.prod(a::AbstractFloat, x::Variable) = Variable(a .* x.data)

function Base.prod(stack::BPStack, a::AbstractFloat, x::Variable)
    y = prod(a, x)
    push!(stack, () -> bwd_prod(y, a, x))
    return y
end

Base.prod(x1::Variable, x2::Variable) = Variable(x1.data .* x2.data)

function Base.prod(stack::BPStack, x1::Variable, x2::Variable)
    y = prod(x1, x2)
    push!(stack, () -> bwd_prod(y, x1, x2))
    return y
end

# -- Division -- #
# function Base.div(stack::BPStack, x1::Variable, x2::Variable)

# -- Linear -- #
function bwd_linear{T,M,K,N}(y::Variable{T,M,N}, w::Variable{T,M,K}, x::Variable{T,K,N})
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

function linear{T,M,K,N}(w::Variable{T,M,K}, x::Variable{T,K,N})
    y = zeros(eltype(w), M, N)
    A_mul_B!(y, w.data, x.data)
    return Variable(y)
end

function linear(stack::BPStack, w, x)
    y = linear(w, x)
    push!(stack, () -> bwd_linear(y, w, x))
    return y
end

# -- Sum -- #
function bwd_sum{T,M,N}(y::Variable{T,M,N}, x1::Variable{T,M,N}, x2::Variable{T,M,N})
    for i in eachindex(y)
        x1.grad[i] += y.grad[i]
        x2.grad[i] += y.grad[i]
    end
    return nothing
end

function bwd_sum{T,M,N}(y::Variable{T,M,N}, x1::Variable{T,M,N}, x2::Variable{T,M,1})
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

Base.sum{T,M,N1,N2}(x1::Variable{T,M,N1}, x2::Variable{T,M,N2}) = Variable(x1.data .+ x2.data)

function Base.sum{T,M,N1,N2}(stack::BPStack, x1::Variable{T,M,N1}, x2::Variable{T,M,N2})
    y = sum(x1, x2)
    push!(stack, () -> bwd_sum(y, x1, x2))
    return y
end

# -- Sum (arbitrary number of blocks) -- #
function Base.sum{T<:AbstractVariable}(xs::Vector{T})
    y = sum(xs[1], xs[2])
    for i = 3:length(xs)
        y = sum(y, xs[i])
    end
    return y
end

function Base.sum{T<:AbstractVariable}(stack::BPStack, xs::Vector{T})
    y = sum(stack, xs[1], xs[2])
    for i = 3:length(xs)
        y = sum(stack, y, xs[i])
    end
    return y
end

Base.sum(x1::Variable, x2::Variable, x3::Variable, xrest::Variable...) = sum([x1, x2, x3, xrest...])

Base.sum(stack::BPStack, x1::Variable, x2::Variable, x3::Variable, xrest::Variable...) = sum(stack, [x1, x2, x3, xrest...])

# -- Minus -- #
function bwd_minus{T,M,N}(y::Variable{T,M,N}, a::AbstractFloat, x::Variable{T,M,N})
    for i in eachindex(y)
        x.grad[i] -= y.grad[i]
    end
    return nothing
end

function bwd_minus{T,M,N}(y::Variable{T,M,N}, x1::Variable{T,M,N}, x2::Variable{T,M,N})
    for i in eachindex(y)
        x1.grad[i] += y.grad[i]
        x2.grad[i] -= y.grad[i]
    end
    return nothing
end

minus(a::AbstractFloat, x::Variable) = Variable(a .- x.data)

function minus(stack::BPStack, a::AbstractFloat, x::Variable)
    y = minus(a, x)
    push!(stack, () -> bwd_minus(y, a, x))
    return y
end

minus{T,M,N}(x1::Variable{T,M,N}, x2::Variable{T,M,N}) = Variable(x1.data .- x2.data)

function minus{T,M,N}(stack::BPStack, x1::Variable{T,M,N}, x2::Variable{T,M,N})
    y = minus(x1, x2)
    push!(stack, () -> bwd_minus(y, x1, x2))
    return y
end

# -- Threshold -- #
function bwd_threshold{T,M,N}(y::Variable{T,M,N}, x::Variable{T,M,N})
    for i = 1:size(x, 1)
        if y.data[i] > 0
            x.grad[i] += y.grad[i]
        end
    end
    return nothing
end

function threshold(x::Variable, t::AbstractFloat)
    data = x.data
    y = similar(data)
    for i in eachindex(x)
        y[i] = data[i] < t ? 0 : 1
    end
    return Variable(y)
end

function threshold(stack::BPStack, x::Variable, t::AbstractFloat)
    y = threshold(x, t)
    push!(stack, () -> bwd_threshold(y, x))
    return y
end

# -- Concatenation --#
function bwd_concat{T<:AbstractVariable}(y::Variable, xs::Vector{T})
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

function concat{T<:AbstractVariable}(xs::Vector{T})
    n2 = size(xs[1], 2)
    for i = 2:length(xs)
        @assert n2 == size(xs[i], 2)
    end
    return Variable(vcat([x.data for x in xs]...))
end

function concat{T<:AbstractVariable}(stack::BPStack, xs::Vector{T})
    y = concat(xs)
    push!(stack, () -> bwd_concat(y, xs))
    return y
end


# -- Deconcatentation -- #
function bwd_decat{T<:AbstractVariable}(y::Vector{T}, x::Variable)
    @assert length(y) == size(x, 1)
    for i = 1:size(x, 1)
        x.grad[i,:] += y[i].grad
    end
    return nothing
end

function decat(x::Variable)
    return Variable[Variable(x.data[i,:]) for i = 1:size(x, 1)]
end

function decat(stack::BPStack, x::Variable)
    y = decat(x)
    push!(stack, () -> bwd_decat(y, x))
    return y
end

# -- Map (single) -- #
# function Base.map{T<:AbstractVariable}(f::Function, xs::Vector{T})
#     ys = Array(T, length(xs))
#     for i = 1:length(xs)
#         ys[i] = f(xs[i])
#     end
#     return ys
# end
#
# function Base.map{T<:AbstractVariable}(stack::BPStack, f::Function, xs::Vector{T})
#     ys = Array(T, length(xs))
#     for i = 1:length(xs)
#         ys[i] = f(stack, xs[i])
#     end
#     return ys
# end

# -- Map (multiple) -- #
# function Base.map{T<:AbstractVariable}(f::Function, xss::Vector{T}...)
#     n = length(xss)
#     len = length(xss[1])
#     for i = 2:n
#         @assert len == length(xss[i])
#     end
#     ys = Array(T, len)
#     for i = 1:len
#         ys[i] = f([xss[k][i] for k = 1:n]...)
#     end
#     return ys
# end
#
# function Base.map{T<:AbstractVariable}(stack::BPStack, f::Function, xss::Vector{T}...)
#     n = length(xss)
#     len = length(xss[1])
#     for i = 2:n
#         @assert len == length(xss[i])
#     end
#     ys = Array(T, len)
#     for i = 1:len
#         ys[i] = f(stack, [xss[k][i] for k = 1:n]...)
#     end
#     return ys
# end

# -- Convenience Functions -- #
affine(w::Variable, x::Variable, b::Variable) = sum(linear(w, x), b)

affine(stack::BPStack, w::Variable, x::Variable, b::Variable) = sum(stack, linear(stack, w, x), b)

# -- Special Ops --#
function dropout!(x::Variable, dp::AbstractFloat)
    data = x.data
    for i in eachindex(x)
        if rand() < dp
            data[i] = 0
        end
    end
    return x
end

dropout!(stack::BPStack, x::Variable, dp::AbstractFloat) = dropout!(x, dp)
