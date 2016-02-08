
type ReverseAffine{Ty<:GradVariable,Tw<:Variable,Tx<:Variable,Tb<:Variable} <: ReverseOperation
    y::Ty
    w::Tw
    x::Tx
    b::Tb
end

@generated function call{Ty,Tw,Tx,Tb}(rop::ReverseAffine{Ty,Tw,Tx,Tb})
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

function affine!(y::AbstractMatrix, w::AbstractMatrix, x::AbstractMatrix, b::AbstractMatrix)
    ysz = size(y)
    wsz = size(w)
    xsz = size(x)
    bsz = size(b)

    bsz == (wsz[1], 1) || throw(OperationError("b must be a $(wsz[1])x1 matrix"))
    ysz == (wsz[1], xsz[2]) || throw(OperationError("y must be a $(wsz[1])x$(xsz[2]) matrix"))

    A_mul_B!(y, w, x)
    for j = 1:xsz[2]
        for i = 1:wsz[1]
            y[i,j] += b[i]
        end
    end
    return y
end

affine(w::AbstractArray, x::AbstractArray, b::AbstractArray) = affine!(zeros(size(w, 1), size(x, 2)), w, x, b)

@generated function affine{Tw<:Variable,Tx<:Variable,Tb<:Variable}(scope::Scope, w::Tw, x::Tx, b::Tb)
    if anygrads(Tw, Tx, Tb) && scope <: GradScope
        return quote
            y_data = allocate(scope, eltype(x), (size(w, 1), size(x, 2)))
            affine!(y_data, w.data, x.data, b.data)
            y = GradVariable(y_data, similar(scope, y_data, 0))
            push_callback!(scope, ReverseAffine(y, w, x, b))
            return y
        end
    else
        return quote
            y_data = allocate(scope, eltype(x), (size(w, 1), size(x, 2)))
            affine!(y_data, w.data, x.data, b.data)
            y = DataVariable(y_data)
            return y
        end
    end
end
