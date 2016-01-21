
type ReverseAffine{Ty<:Variable,Tw<:Variable,Tx<:Variable,Tb<:Variable} <: ReverseOperation
    y::Ty
    w::Tw
    x::Tx
    b::Tb
end

call{Ty<:DataVariable,Tw,Tx,Tb}(rop::ReverseAffine{Ty,Tw,Tx,Tb}) = nothing

@generated function call{Ty<:GradVariable,Tw,Tx,Tb}(rop::ReverseAffine{Ty,Tw,Tx,Tb})
    stmts = Any[
        :(y = rop.y),
        :(w = rop.w),
        :(x = rop.x),
        :(b = rop.b),
    ]
    if Tw <: GradVariable
        s = quote
            dw = A_mul_Bt(y.grad, x.data)
            for i in eachindex(w)
                w.grad[i] += dw[i]
            end
        end
        push!(stmts, s)
    end
    if Tx <: GradVariable
        s = quote
            dx = At_mul_B(w.data, y.grad)
            for i in eachindex(x)
                x.grad[i] += dx[i]
            end
        end
        push!(stmts, s)
    end
    if Tb <: GradVariable
        s = quote
            for j = 1:size(y, 2)
                for i = 1:size(y, 1)
                    b.grad[i] += y.grad[i,j]
                end
            end
        end
        push!(stmts, s)
    end
    push!(stmts, :(return nothing))
    return Expr(:block, stmts...)
end

@generated function affine{Tw<:Variable,Tx<:Variable,Tb<:Variable}(w::Tw, x::Tx, b::Tb)
    rtype = if Tw <: GradVariable || Tx <: GradVariable || Tb <: GradVariable
        GradVariable
    else
        DataVariable
    end
    return quote
        size(b) == (size(w, 1), 1) || throw(OperationError("b must be a $(size(w, 1))x1 matrix"))
        tmp = w.data * x.data
        bias = b.data
        for j = 1:size(tmp, 2)
            for i = 1:size(tmp, 1)
                tmp[i,j] += bias[i]
            end
        end
        return $rtype(tmp)
    end
end

function affine(stack::CallbackStack, w::Variable, x::Variable, b::Variable)
    y = affine(w, x, b)
    push_callback!(stack, ReverseAffine(y, w, x, b))
    return y
end
