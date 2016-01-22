
type ReversePlus{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

# call{T<:DataVariable}(rop::ReverseSum{T}) = nothing

function call(rop::ReversePlus)
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] += y.grad[i]
    end
    return nothing
end

type ReverseColumnPlus{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

# call{T<:DataVariable}(rop::ReverseColumnSum{T}) = nothing

function call(rop::ReverseColumnPlus)
    y = rop.y
    x = rop.x
    for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            x.grad[i] += y.grad[i,j]
        end
    end
    return nothing
end

plus(a::AbstractMatrix, b::AbstractMatrix) = a .+ b

plus(a::Variable, b::Variable) = DataVariable(plus(a.data, b.data))

@generated function plus{Ta<:Variable,Tb<:Variable}(stack::CallbackStack, a::Ta, b::Tb)
    if anygrads(Ta, Tb)
        return quote
            c = GradVariable(plus(a.data, b.data))
            if size(a) == size(b)
                isa(a, GradVariable) && push!(stack, ReversePlus(c, a))
                isa(b, GradVariable) && push!(stack, ReversePlus(c, b))
            elseif is_matrix(a) && is_column_vector(b)
                isa(a, GradVariable) && push!(stack, ReversePlus(c, a))
                isa(b, GradVariable) && push!(stack, ReverseColumnPlus(c, b))
            elseif is_matrix(b) && is_column_vector(a)
                isa(b, GradVariable) && push!(stack, ReversePlus(c, b))
                isa(a, GradVariable) && push!(stack, ReverseColumnPlus(c, a))
            else
                throw(OperationError("no plus for sizes a: $(size(a)), b: $(size(b))"))
            end
            return c
        end
    else
        return :(plus(a, b))
    end
end

# -- Plus > 2 -- #
function plus{V<:Variable}(xs::Vector{V})
    y = plus(xs[1], xs[2])
    for i = 3:length(xs)
        y = plus(y, xs[i])
    end
    return y
end

function plus{V<:Variable}(stack::CallbackStack, xs::Vector{V})
    y = plus(stack, xs[1], xs[2])
    for i = 3:length(xs)
        y = plus(stack, y, xs[i])
    end
    return y
end

plus(x1::Variable, x2::Variable, x3::Variable, xrest::Variable...) = plus([x1, x2, x3, xrest...])

plus(stack::CallbackStack, x1::Variable, x2::Variable, x3::Variable, xrest::Variable...) = plus(stack, [x1, x2, x3, xrest...])

