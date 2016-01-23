"""
Reverse op for elementwise minus.
if      c = minus(a, b) 
where   size(a) == size(b)
then    c[i] = a[i] - b[i].

Gradient propagation
    da[i] += dc[i]
    db[i] -= dc[i]
"""
type ReverseMinus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseMinus{Tc,Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i] += c.grad[i]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i] -= c.grad[i]))
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
Reverse op for elementwise minus broadcast over rows.
if      c = minus(a, b)
where   size(a) == (1, size(b, 2))
then    c[i,j] = a[j] - b[i,j]

Gradient propagation
    da[j] += dc[i,j]
    db[i,j] -= db[i,j]
"""
type ReverseRowBroadcastMinus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseRowBroadcastMinus{Tc,Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[j] += c.grad[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] -= c.grad[i,j]))
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
Reverse op for elementwise minus broadcast over columns.
if      c = minus(a, b)
where   size(a) == (size(b, 1), 1)
then    c[i,j] = a[i] - b[i,j]

Gradient propagation
    da[i] += dc[i,j]
    db[i,j] -= db[i,j]
"""
type ReverseColBroadcastMinus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseColBroadcastMinus{Tc,Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i] += c.grad[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] -= c.grad[i,j]))
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

minus(a::AbstractArray, b::AbstractArray) = a .- b

minus(a::Variable, b::Variable) = DataVariable(minus(a.data, b.data))

@generated function minus{Ta<:Variable,Tb<:Variable}(stack::CallbackStack, a::Ta, b::Tb)
    if anygrads(Ta, Tb)
        return quote
            c = GradVariable(minus(a.data, b.data))
            asz, bsz = size(a), size(b)
            if asz == bsz
                push!(stack, ReverseMinus(c, a, b))
            elseif asz == (1, bsz[2])
                push!(stack, ReverseRowBroadcastMinus(c, a, b))
            elseif asz == (bsz[1], 1)
                push!(stack, ReverseColBroadcastMinus(c, a, b))
            else
                throw(OperationError("no minus for sizes a: $(size(a)), b: $(size(b))"))
            end
            return c
        end
    else
        return :(minus(a, b))
    end
end
