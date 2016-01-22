
type ReverseMinus{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseMinus)
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] -= y.grad[i]
    end
    return nothing
end

type ReverseColumnMinus{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseColumnMinus)
    y = rop.y
    x = rop.x
    for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            x.grad[i] -= y.grad[i,j]
        end
    end
    return nothing
end

minus(a::AbstractArray, b::AbstractArray) = a .- b

minus(a::Variable, b::Variable) = DataVariable(minus(a.data, b.data))

@generated function minus{Ta<:Variable,Tb<:Variable}(stack::CallbackStack, a::Ta, b::Tb)
    if anygrads(Ta, Tb)
        return quote
            c = GradVariable(minus(a.data, b.data))
            if size(a) == size(b)
                isa(a, GradVariable) && push!(stack, ReversePlus(c, a))
                isa(b, GradVariable) && push!(stack, ReverseMinus(c, b))
            elseif is_matrix(a) && is_column_vector(b)
                isa(a, GradVariable) && push!(stack, ReversePlus(c, a))
                isa(b, GradVariable) && push!(stack, ReverseColumnMinus(c, b))
            elseif is_matrix(b) && is_column_vector(a)
                isa(b, GradVariable) && push!(stack, ReverseMinus(c, b))
                isa(a, GradVariable) && push!(stack, ReverseColumnPlus(c, a))
            else
                throw(OperationError("no minus for sizes a: $(size(a)), b: $(size(b))"))
            end
            return c
        end
    else
        return :(minus(a, b))
    end
end
