
type ReverseRelu{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseRelu)
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

relu(stack::CallbackStack, x::DataVariable) = DataVariable(relu(x.data))

function relu(stack::CallbackStack, x::GradVariable)
    y = GradVariable(relu(x.data))
    push!(stack, ReverseRelu(y, x))
    return y
end
