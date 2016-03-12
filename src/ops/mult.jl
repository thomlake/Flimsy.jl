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
        @inbounds for i in eachindex(c)
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
        @inbounds for j = 1:size(c, 2)
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
        @inbounds for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $inner
            end
        end
        return nothing
    end
end

function mult_elementwise!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @inbounds for i in eachindex(a)
        c[i] = a[i] * b[i]
    end
    return c
end

function mult_row_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @inbounds for j = 1:size(b, 2)
        for i = 1:size(b, 1)
            c[i,j] = a[1,j] * b[i,j]
        end
    end
    return c
end

function mult_column_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @inbounds for j = 1:size(b, 2)
        for i = 1:size(b, 1)
            c[i,j] = a[i] * b[i,j]
        end
    end
    return c
end

mult(a::AbstractArray, b::AbstractArray) = a .* b

mult(a::Variable, b::Variable) = DataVariable(mult(a.data, b.data))

@generated function mult{Ta<:Variable,Tb<:Variable}(scope::Scope, a::Ta, b::Tb)
    if anygrads(Ta, Tb) && scope <: GradScope
        return quote
            asz, bsz = size(a), size(b)
            if asz == bsz
                c_data = similar(scope, b.data)
                mult_elementwise!(c_data, a.data, b.data)
                c = GradVariable(c_data, similar(scope, c_data, 0))
                push_callback!(scope, ReverseMult(c, a, b))
                return c
            elseif asz == (1, bsz[2])
                c_data = similar(scope, b.data)
                mult_row_broadcast!(c_data, a.data, b.data)
                c = GradVariable(c_data, similar(scope, c_data, 0))
                push_callback!(scope, ReverseRowBroadcastMult(c, a, b))
                return c
            elseif asz == (bsz[1], 1)
                c_data = similar(scope, b.data)
                mult_column_broadcast!(c_data, a.data, b.data)
                c = GradVariable(c_data, similar(scope, c_data, 0))
                push_callback!(scope, ReverseColBroadcastMult(c, a, b))
                return c
            elseif bsz == (1, asz[2])
                c_data = similar(scope, a.data)
                mult_row_broadcast!(c_data, b.data, a.data)
                c = GradVariable(c_data, similar(scope, c_data, 0))
                push_callback!(scope, ReverseRowBroadcastMult(c, b, a))
                return c
            elseif bsz == (asz[1], 1)
                c_data = similar(scope, a.data)
                mult_column_broadcast!(c_data, b.data, a.data)
                c = GradVariable(c_data, similar(scope, c_data, 0))
                push_callback!(scope, ReverseColBroadcastMult(c, b, a))
                return c
            else
                throw(OperationError("no mult for sizes a: $(size(a)), b: $(size(b))"))
            end
        end
    else
        return quote
            asz, bsz = size(a), size(b)
            if asz == bsz
                c_data = similar(scope, b.data)
                mult_elementwise!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            elseif asz == (1, bsz[2])
                c_data = similar(scope, b.data)
                mult_row_broadcast!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            elseif asz == (bsz[1], 1)
                c_data = similar(scope, b.data)
                mult_column_broadcast!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            elseif bsz == (1, asz[2])
                c_data = similar(scope, a.data)
                mult_row_broadcast!(c_data, b.data, a.data)
                c = DataVariable(c_data)
                return c
            elseif bsz == (asz[1], 1)
                c_data = similar(scope, a.data)
                mult_column_broadcast!(c_data, b.data, a.data)
                c = DataVariable(c_data)
                return c
            else
                throw(OperationError("no mult for sizes a: $(size(a)), b: $(size(b))"))
            end
        end
    end
end
