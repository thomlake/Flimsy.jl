type ReverseScalarMinusMatrix{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseScalarMinusMatrix{T}) = nothing

function call{T<:GradVariable}(rop::ReverseScalarMinusMatrix{T})
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] -= y.grad[i]
    end
    return nothing
end

minus{V<:Variable}(a::Real, x::V) = V(a .- x.data)

function minus(stack::CallbackStack, a::Real, x::Variable)
    y = minus(a, x)
    push_callback!(stack, ReverseScalarMinusMatrix(y, x))
    return y
end

type ReverseMatrixMinusScalar{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseMatrixMinusScalar{T}) = nothing

function call{T<:GradVariable}(rop::ReverseMatrixMinusScalar{T})
    y = rop.y
    x = rop.x
    for i in eachindex(y)
        x.grad[i] += y.grad[i]
    end
    return nothing
end

minus{V<:Variable}(x::V, a::Real) = V(x.data .- a)

function minus(stack::CallbackStack, x::Variable, a::Real)
    y = minus(x, a)
    push_callback!(stack, ReverseMatrixMinusScalar(y, x))
    return y
end
