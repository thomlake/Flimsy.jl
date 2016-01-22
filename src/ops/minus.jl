
type ReverseMinus{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseMinus{T}) = nothing

function call{T<:GradVariable}(rop::ReverseMinus{T})
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] -= y.grad[i]
    end
    return nothing
end

type ReverseColumnMinus{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseColumnMinus{T}) = nothing

function call{T<:GradVariable}(rop::ReverseColumnMinus{T})
    y = rop.y
    x = rop.x
    for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            x.grad[i] -= y.grad[i,j]
        end
    end
    return nothing
end

minus(a::Variable, b::Variable) = DataVariable(a.data .- b.data) 

@generated function minus{Ta<:Variable,Tb<:Variable}(stack::CallbackStack, a::Ta, b::Tb)
    if Ta <: GradVariable || Tb <: GradVariable
        return quote
            c = GradVariable(a.data .- b.data)
            if size(a) == size(b)
                push_callback!(stack, ReverseSum(c, a))
                push_callback!(stack, ReverseMinus(c, b))
            elseif is_matrix(a) && is_column_vector(b)
                push_callback!(stack, ReverseSum(c, a))
                push_callback!(stack, ReverseColumnMinus(c, b))
            elseif is_matrix(b) && is_column_vector(a)
                push_callback!(stack, ReverseMinus(c, b))
                push_callback!(stack, ReverseColumnSum(c, a))
            else
                error("no minus for sizes a: $(size(a)), b: $(size(b))")
            end
            return c
        end
    else
        return :(DataVariable(a.data .- b.data))
    end
end
