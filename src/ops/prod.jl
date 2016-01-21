
type ReverseBroadcastProd{Tc<:Variable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseBroadcastProd{Tc,Ta,Tb})
    if Tc <: DataVariable
        return
    end
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[j] += c.grad[i,j] * b.data[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] += c.grad[i,j] * a.data[j]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $inner
            end
        end
        return nothing
    end
end

type ReverseProd{Tc<:Variable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseProd{Tc,Ta,Tb})
    if Tc <: DataVariable
        return
    end
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i,j] += c.grad[i,j] * b.data[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] += c.grad[i,j] * a.data[i,j]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $inner
            end
        end
        return nothing
    end
end

@generated function Base.prod{Va<:Variable,Vb<:Variable}(a::Va, b::Vb)
    if Va <: GradVariable || Vb <: GradVariable
        return :(GradVariable(a.data .* b.data))
    else
        return :(DataVariable(a.data .* b.data))
    end
end

function Base.prod(stack::CallbackStack, a::Variable, b::Variable)
    c = prod(a, b)
    if size(a) == size(b)
        push_callback!(stack, ReverseProd(c, a, b))
    elseif is_matrix(a) && is_row_vector(b)
        push_callback!(stack, ReverseBroadcastProd(c, b, a))
    elseif is_row_vector(a) && is_matrix(b)
        push_callback!(stack, ReverseBroadcastProd(c, a, b))
    elseif is_column_vector(a) && is_scalar(b)
        push_callback!(stack, ReverseBroadcastProd(c, b, a))
    elseif is_scalar(a) && is_column_vector(b)
        push_callback!(stack, ReverseBroadcastProd(c, a, b))
    else
        error("no prod for sizes a: $(size(a)), b: $(size(b))")
    end
    return c
end
