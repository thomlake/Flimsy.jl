
type ReverseWTA{T<:GradVariable} <: ReverseOperation
    y::T
    x::T
end

function call(rop::ReverseWTA)
    y = rop.y
    x = rop.x
    _, imax = findmax(x.data, 1)
    for i = 1:endof(imax)
        x.grad[imax[i]] += y.grad[imax[i]]
    end
    return nothing
end

Base.tanh{V<:Variable}(x::V) = V(tanh(x.data))

function wta(x::Variable)
    y = zero(x)
    xmax, imax = findmax(x.data, 1)
    for i = 1:endof(imax)
        y.data[imax[i]] = xmax[i]
    end
    return y
end

function wta(stack::CallbackStack, x::GradVariable)
    y = wta(x)
    push_callback!(stack, ReverseWTA(y, x))
    return y
end
