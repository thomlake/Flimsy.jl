
type ReverseRelu{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseRelu{T}) = nothing

function call{T<:GradVariable}(rop::ReverseRelu{T})
    y = rop.y
    x = rop.x
    for i in eachindex(x)
        if x.data[i] > 0
            x.grad[i] += y.grad[i]
        end
    end
    return nothing
end

relu(x::AbstractArray) = max(0, x)

relu(x::Variable) = DataVariable(relu(x.data))

function relu(stack::CallbackStack, x::GradVariable)
    y = GradVariable(relu(x.data))
    push_callback!(stack, ReverseRelu(y, x))
    return y
end

relu(stack::CallbackStack, x::DataVariable) = GradVariable(relu(x.data))
