
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

relu{V<:Variable}(x::V) = V(relu(x.data))

function relu(stack::CallbackStack, x::Variable)
    y = relu(x)
    push_callback!(stack, ReverseRelu(y, x))
    return y
end
