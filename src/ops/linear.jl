
type ReverseLinear{Ty<:GradVariable,Tw<:Variable,Tx<:Variable} <: ReverseOperation
    y::Ty
    w::Tw
    x::Tx
end


@generated function call{Ty,Tw,Tx}(rop::ReverseLinear{Ty,Tw,Tx})
    stmts = Any[
        :(y = rop.y),
        :(w = rop.w),
        :(x = rop.x),
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

    push!(stmts, :(return nothing))
    return Expr(:block, stmts...)
end

linear!(y::AbstractArray, w::AbstractArray, x::AbstractArray) = A_mul_B!(y, w, x)

linear(w::AbstractArray, x::AbstractArray) = w * x

# linear(w::Variable, x::Variable) = DataVariable(linear(w.data, x.data))

@generated function linear{Tw<:Variable,Tx<:Variable}(scope::Scope, w::Tw, x::Tx)
    if anygrads(Tw, Tx) && scope <: GradScope
        return quote
            y_data = allocate(scope, eltype(x.data), (size(w, 1), size(x, 2)))
            linear!(y_data, w.data, x.data)
            y = GradVariable(y_data, similar(scope, y_data, 0))
            push_callback!(scope, ReverseLinear(y, w, x))
            return y
        end
    else
        return quote
            y_data = allocate(scope, eltype(x.data), (size(w, 1), size(x, 2)))
            linear!(y_data, w.data, x.data)
            return DataVariable(y_data)
        end
    end
end
