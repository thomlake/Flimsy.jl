
type ReverseBroadcastMult{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseBroadcastMult{Tc,Ta,Tb})
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

type ReverseMult{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseMult{Tc,Ta,Tb})
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

mult(a::AbstractArray, b::AbstractArray) = a .* b

mult(a::Variable, b::Variable) = DataVariable(mult(a.data, b.data))

@generated function mult{Ta<:Variable,Tb<:Variable}(stack::CallbackStack, a::Ta, b::Tb)
    if anygrads(Ta, Tb)
        return quote
            c = GradVariable(mult(a.data, b.data))
            if size(a) == size(b)
                push!(stack, ReverseMult(c, a, b))
            elseif is_matrix(a) && is_row_vector(b)
                push!(stack, ReverseBroadcastMult(c, b, a))
            elseif is_row_vector(a) && is_matrix(b)
                push!(stack, ReverseBroadcastMult(c, a, b))
            elseif is_column_vector(a) && is_scalar(b)
                push!(stack, ReverseBroadcastMult(c, b, a))
            elseif is_scalar(a) && is_column_vector(b)
                push!(stack, ReverseBroadcastMult(c, a, b))
            else
                throw(OperationError("no prod for sizes a: $(size(a)), b: $(size(b))"))
            end
            return c
        end
    else
        return :(mult(a, b))
    end
end
