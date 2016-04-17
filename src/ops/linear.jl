
type ReverseLinear{Tw<:Variable,Tx<:Variable} <: ReverseOperation
    y::GradVariable
    w::Tw
    x::Tx
end


@generated function call{Tw,Tx}(rop::ReverseLinear{Tw,Tx})
    stmts = Any[
        :(y = rop.y),
        :(w = rop.w),
        :(x = rop.x),
    ]
    
    if Tw <: GradVariable
        push!(stmts, :(add_to_A_mul_Bt!(w.grad, y.grad, x.data)))
    end
    
    if Tx <: GradVariable
        push!(stmts, :(add_to_At_mul_B!(x.grad, w.data, y.grad)))
    end

    push!(stmts, :(return nothing))
    return Expr(:block, stmts...)
end

linear!(y::AbstractArray, w::AbstractArray, x::AbstractArray) = A_mul_B!(y, w, x)

linear(w::AbstractArray, x::AbstractArray) = w * x

@generated function linear{Tw<:Variable,Tx<:Variable}(scope::Scope, w::Tw, x::Tx)
    if anygrads(Tw, Tx) && scope <: GradScope
        return quote
            y_data = Matrix{FloatX}(size(w, 1), size(x, 2))
            linear!(y_data, w.data, x.data)
            y = GradVariable(y_data, zero(y_data))
            push_callback!(scope, ReverseLinear(y, w, x))
            return y
        end
    else
        return quote
            y_data = Matrix{FloatX}(size(w, 1), size(x, 2))
            linear!(y_data, w.data, x.data)
            return DataVariable(y_data)
        end
    end
end
