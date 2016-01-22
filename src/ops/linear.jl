
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

linear(w::AbstractArray, x::AbstractArray) = w * x

linear(w::Variable, x::Variable) = DataVariable(linear(w.data, x.data))

@generated function linear{Tw<:Variable,Tx<:Variable}(stack::CallbackStack, w::Tw, x::Tx)
    stmts = Any[]
    if anygrads(Tw, Tx)
        return quote
            y = GradVariable(linear(w.data, x.data))
            push!(stack, ReverseLinear(y, w, x))
            return y
        end
    else
        return :(linear(w, x))
    end
end
