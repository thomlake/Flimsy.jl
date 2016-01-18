
type ReverseLinear{T<:GradVariable} <: ReverseOperation
    y::T
    w::T
    x::T
end

function call(rop::ReverseLinear)
    y = rop.y
    w = rop.w
    x = rop.x
    dx = At_mul_B(w.data, y.grad)
    dw = A_mul_Bt(y.grad, x.data)
    for i in eachindex(x)
        x.grad[i] += dx[i]
    end
    for i in eachindex(w)
        w.grad[i] += dw[i]
    end
    return nothing
end

linear{V1<:Variable,V2<:Variable}(w::V1, x::V2) = V2(w.data * x.data)

function linear(stack::CallbackStack, w::GradVariable, x::GradVariable)
    y = linear(w, x)
    push_callback!(stack, ReverseLinear(y, w, x))
    return y
end
