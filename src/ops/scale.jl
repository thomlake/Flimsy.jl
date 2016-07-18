"""
Scale by a primitave (float, int, literal, etc)
"""
type ReverseScalePrimative{T<:Real} <: ReverseOperation
    c::Variable
    a::T
    b::Variable
end

function call(rop::ReverseScalePrimative)
    c = rop.c
    a = rop.a
    b = rop.b
    @flimsy_inbounds for i in eachindex(c)
        b.grad[i] += a * c.grad[i]
    end
    return nothing
end

Base.scale(scope::Scope, a::Real, b::AbstractValue) = Constant(scale(a, b.data))

function Base.scale(scope::GradScope, a::Real, b::Variable)
    c = Variable(scale(a, b.data))
    push_callback!(scope, ReverseScalePrimative(c, a, b))
    return c
end

"""
Scale
"""
type ReverseScale{A<:AbstractValue,B<:AbstractValue} <: ReverseOperation
    c::Variable
    a::A
    b::B
end

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    updates = Any[]
    A <: Variable && push!(updates, :(a.grad[j] += b.data[i,j] * c.grad[i,j]))
    B <: Variable && push!(updates, :(b.grad[i,j] += a.data[j] * c.grad[i,j]))
    update_block = Expr(:block, updates...)
    defn = :(function call(rop::ReverseScale{$A,$B})
        c = rop.c
        a = rop.a
        b = rop.b
        @flimsy_inbounds for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $update_block
            end
        end
        nothing
    end)
    eval(defn)
end

function inplace_scale!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    size(a) == (1, size(b, 2)) || throw(DimensionMismatch("size(a) = $(size(a)) != (1, $(size(b, 2))) = (1, size(b, 2))"))
    @flimsy_inbounds for j = 1:size(c, 2)
        for i = 1:size(c, 1)
            c[i,j] = a[j] * b[i,j]
        end
    end
    return c
end

Base.scale(scope::Scope, a::AbstractValue, b::AbstractValue) = Constant(inplace_scale!(similar(b.data), a.data, b.data))

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    defn = :(function Base.scale(scope::GradScope, a::$A, b::$B)
        c = Variable(inplace_scale!(similar(b.data), a.data, b.data))
        push_callback!(scope, ReverseScale(c, a, b))
        return c
    end)
    eval(defn)
end
