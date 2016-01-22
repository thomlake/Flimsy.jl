
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

linear(w::Variable, x::Variable) = DataVariable(w.data * x.data)

@generated function linear{Tw<:Variable,Tx<:Variable}(stack::CallbackStack, w::Tw, x::Tx)
    stmts = Any[]
    if Tw <: GradVariable || Tx <: GradVariable
        s = quote
            y = GradVariable(w.data * x.data)
            push!(stack, ReverseLinear(y, w, x))
        end
        push!(stmts, s)
    else
        push!(stmts, :(y = DataVariable(w.data * x.data)))
    end
    push!(stmts, :(return y))
    return Expr(:block, stmts...)
end
