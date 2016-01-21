
type ReverseAffine{Ty<:Variable,Tw<:Variable,Tx<:Variable,Tb<:Variable} <: ReverseOperation
    y::Ty
    w::Tw
    x::Tx
    b::Tb
end

@generated function call{Ty,Tw,Tx,Tb}(rop::ReverseAffine{Ty,Tw,Tx,Tb})
    y = rop.y
    w = rop.w
    x = rop.x
    b = rop.b
    dx = At_mul_B(w.data, y.grad)
    dw = A_mul_Bt(y.grad, x.data)
    for i in eachindex(x)
        x.grad[i] += dx[i]
    end
    for i in eachindex(w)
        w.grad[i] += dw[i]
    end
    for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            b.grad[i] += y.grad[i,j]
        end
    end
    return nothing
end

function affine{V<:Variable}(w::V, x::V, b::V)
    size(b) == (size(w, 1), 1) || throw(OperationError("b must be a $(size(w, 1))x1 matrix"))
    tmp = w.data * x.data
    bias = b.data
    for j = 1:size(tmp, 2)
        for i = 1:size(tmp, 1)
            tmp[i,j] += bias[i]
        end
    end
    return V(tmp)
end

function affine(stack::CallbackStack, w::GradVariable, x::GradVariable, b::GradVariable)
    y = affine(w, x, b)
    push_callback!(stack, ReverseAffine(y, w, x, b))
    return y
end
