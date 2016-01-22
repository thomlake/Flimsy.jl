
type ReverseConcat{T<:Variable} <: ReverseOperation
    y::T
    xs::Vector{T}
end

call{T<:DataVariable}(rop::ReverseConcat{T}) = nothing

function call{T<:GradVariable}(rop::ReverseConcat{T})
    y = rop.y
    xs = rop.xs
    offset = 0
    for k = 1:length(xs)
        for j = 1:size(xs[k], 2)
            for i = 1:size(xs[k], 1)
                xs[k].grad[i,j] += y.grad[offset + i,j]
            end
        end
        offset += size(xs[k], 1)
    end
    return nothing
end

function concat{V<:Variable}(xs::Vector{V})
    n = size(xs[1], 2)
    for i = 2:length(xs)
        n == size(xs[i], 2) || error("can only concatenate vectors with members having the same number of columns")
    end
    return DataVariable(vcat([x.data for x in xs]...))
end

function concat{T<:GradVariable}(stack::CallbackStack, xs::Vector{T})
    n = size(xs[1], 2)
    for i = 2:length(xs)
        n == size(xs[i], 2) || error("can only concatenate vectors with members having the same number of columns")
    end
    y = GradVariable(vcat([x.data for x in xs]...))
    push_callback!(stack, ReverseConcat(y, xs))
    return y
end

concat{T<:DataVariable}(stack::CallbackStack, xs::Vector{T}) = concat(xs)

concat{V<:Variable}(xs::V...) = concat([x for x in xs])

concat{V<:Variable}(stack::CallbackStack, xs::V...) = concat(stack, [x for x in xs])
