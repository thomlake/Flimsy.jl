
type ReverseLinear{Ty<:Variable,Tw<:Variable,Tx<:Variable} <: ReverseOperation
    y::Ty
    w::Tw
    x::Tx
end

call{Ty<:DataVariable,Tw,Tx}(rop::ReverseLinear{Ty,Tw,Tx}) = nothing

@generated function call{Ty<:GradVariable,Tw,Tx}(rop::ReverseLinear{Ty,Tw,Tx})
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

@generated function linear{Tw<:Variable,Tx<:Variable}(w::Tw, x::Tx)
    if Tw <: GradVariable || Tx <: GradVariable
        return :(GradVariable(w.data * x.data))
    else
        return :(DataVariable(w.data * x.data))
    end
end

function linear(stack::CallbackStack, w::Variable, x::Variable)
    y = linear(w, x)
    push!(stack, ReverseLinear(y, w, x))
    return y
end
