
type ReverseWTA{T<:Variable} <: ReverseOperation
    y::T
    x::T
end

call{T<:DataVariable}(rop::ReverseWTA{T}) = nothing

function call{T<:GradVariable}(rop::ReverseWTA{T})
    y = rop.y
    x = rop.x
    _, imax = findmax(x.data, 1)
    for i = 1:endof(imax)
        x.grad[imax[i]] += y.grad[imax[i]]
    end
    return nothing
end

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
