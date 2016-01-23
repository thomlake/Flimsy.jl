"""
Reverse op for elementwise multiplication.
if      c = mult(a, b) 
where   size(a) == size(b)
then    c[i] = a[i] * b[i].

Gradient propagation
    da[i] += dc[i] * b[i]
    db[i] += dc[i] * a[i]
"""
type ReverseMult{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseMult{Tc,Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i] += c.grad[i] * b.data[i]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i] += c.grad[i] * a.data[i]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        for i in eachindex(c)
            $inner
        end
        return nothing
    end
end

"""
Reverse op for elementwise multiplication broadcast over rows.
if      c = mult(a, b)
where   size(a) == (1, size(b, 2))
then    c[i,j] = a[j] * b[i,j]

Gradient propagation
    da[j] += dc[i,j] * b[i,j]
    db[i,j] += db[i,j] * a[1,j]
"""
type ReverseRowBroadcastMult{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseRowBroadcastMult{Tc,Ta,Tb})
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

"""
Reverse op for elementwise multiplication broadcast over columns.
if      c = mult(a, b)
where   size(a) == (size(b, 1), 1)
then    c[i,j] = a[i] * b[i,j]

Gradient propagation
    da[i] += dc[i,j] * b[i,j]
    db[i,j] += db[i,j] * a[i]
"""
type ReverseColBroadcastMult{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseColBroadcastMult{Tc,Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i] += c.grad[i,j] * b.data[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] += c.grad[i,j] * a.data[i]))
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
            asz, bsz = size(a), size(b)
            if asz == bsz
                push!(stack, ReverseMult(c, a, b))
            elseif asz == (1, bsz[2])
                push!(stack, ReverseRowBroadcastMult(c, a, b))
            elseif asz == (bsz[1], 1)
                push!(stack, ReverseColBroadcastMult(c, a, b))
            elseif bsz == (1, asz[2])
                push!(stack, ReverseRowBroadcastMult(c, b, a))
            elseif bsz == (asz[1], 1)
                push!(stack, ReverseColBroadcastMult(c, b, a))
            else
                throw(OperationError("no mult for sizes a: $(size(a)), b: $(size(b))"))
            end
            return c
        end
    else
        return :(mult(a, b))
    end
end
