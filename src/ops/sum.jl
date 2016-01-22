
type ReverseSum{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

# call{T<:DataVariable}(rop::ReverseSum{T}) = nothing

function call(rop::ReverseSum{T})
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] += y.grad[i]
    end
    return nothing
end

type ReverseColumnSum{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseColumnSum{T}) = nothing

function call{T<:GradVariable}(rop::ReverseColumnSum{T})
    y = rop.y
    x = rop.x
    for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            x.grad[i] += y.grad[i,j]
        end
    end
    return nothing
end

Base.sum(a::Variable, b::Variable) = DataVariable(a.data .+ b.data)

@generated function Base.sum{Ta<:Variable,Tb<:Variable}(stack::CallbackStack, a::Ta, b::Tb)
    if Ta <: GradVariable || Tb <: GradVariable
        quote
            c = GradVariable(a.data .+ b.data)
            if size(a) == size(b)
                push_callback!(stack, ReverseSum(c, a))
                push_callback!(stack, ReverseSum(c, b))
            elseif is_matrix(a) && is_column_vector(b)
                push_callback!(stack, ReverseSum(c, a))
                push_callback!(stack, ReverseColumnSum(c, b))
            elseif is_matrix(b) && is_column_vector(a)
                push_callback!(stack, ReverseSum(c, b))
                push_callback!(stack, ReverseColumnSum(c, a))
            else
                error("no sum for sizes a: $(size(a)), b: $(size(b))")
            end
            return c
        end
    else
        :(DataVariable(a.data .+ b.data))
    end
end

# # -- Sum (arbitrary number of blocks) -- #
function Base.sum{V<:Variable}(xs::Vector{V})
    y = sum(xs[1], xs[2])
    for i = 3:length(xs)
        y = sum(y, xs[i])
    end
    return y
end

function Base.sum{V<:Variable}(stack::CallbackStack, xs::Vector{V})
    y = sum(stack, xs[1], xs[2])
    for i = 3:length(xs)
        y = sum(stack, y, xs[i])
    end
    return y
end

Base.sum{V<:Variable}(x1::V, x2::V, x3::V, xrest::V...) = sum([x1, x2, x3, xrest...])

Base.sum(stack::CallbackStack, x1::Variable, x2::Variable, x3::Variable, xrest::Variable...) = sum(stack, [x1, x2, x3, xrest...])

