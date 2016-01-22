
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


function affine(w::AbstractMatrix, x::AbstractMatrix, b::AbstractMatrix)
    size(b) == (size(w, 1), 1) || throw(OperationError("b must be a $(size(w, 1))x1 matrix"))
    tmp = w * x
    for j = 1:size(tmp, 2)
        for i = 1:size(tmp, 1)
            tmp[i,j] += b[i]
        end
    end
    return tmp
end

affine(w::Variable, x::Variable, b::Variable) = DataVariable(affine(w.data, x.data, b.data))

@generated function affine{Tw<:Variable,Tx<:Variable,Tb<:Variable}(stack::CallbackStack, w::Tw, x::Tx, b::Tb)
    if anygrads(Tw, Tx, Tb)
        return quote
            y = GradVariable(affine(w.data, x.data, b.data))
            push_callback!(stack, ReverseAffine(y, w, x, b))
            return y
        end
    else
        return :(affine(w, x, b))
    end
end
