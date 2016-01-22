
# minus(Real, Variable) 
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

minus(a::Real, x::Array) = a .- x

minus(a::Real, x::Variable) = DataVariable(minus(a, x.data))

function minus(stack::CallbackStack, a::Real, x::GradVariable)
    y = GradVariable(minus(a, x.data))
    push_callback!(stack, ReverseScalarMinusMatrix(y, x))
    return y
end

minus(stack::CallbackStack, a::Real, x::DataVariable) = DataVariable(minus(a, x.data))

# minus(Variable, Real)
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

minus(x::Array, a::Real) = x .- a

minus(x::Variable, a::Real) = DataVariable(minus(x.data, a))

function minus(stack::CallbackStack, x::GradVariable, a::Real)
    y = GradVariable(minus(x.data, a))
    push_callback!(stack, ReverseMatrixMinusScalar(y, x))
    return y
end

function minus(stack::CallbackStack, x::DataVariable, a::Real)
    y = DataVariable(minus(x.data, a))
    push_callback!(stack, ReverseMatrixMinusScalar(y, x))
    return y
end

