"""
Reverse op for elementwise plus broadcast over columns.
if      c = plus(a, b)
where   size(b) == (size(a, 1), 1)
then    c[i,j] = a[i,j] + b[i]

Gradient propagation
    da[i,j] += dc[i,j]
    db[i] += dc[i,j]
"""
type ReversePlusCols{A<:AbstractValue,B<:AbstractValue} <: ReverseOperation
    c::Variable
    a::A
    b::B
end

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    updates = Any[]
    A <: Variable && push!(updates, :(a.grad[i,j] += c.grad[i,j]))
    B <: Variable && push!(updates, :(b.grad[i] += c.grad[i,j]))
    update_block = Expr(:block, updates...)
    defn = :(function call(rop::ReversePlusCols{$A,$B})
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

function plus_cols!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    size(b) == (size(a, 1), 1) || throw(DimensionMismatch("b must be $(size(a, 1)) by 1, got $(size(b))"))
    @flimsy_inbounds for j = 1:size(c, 2)
        for i = 1:size(c, 1)
            c[i,j] = a[i,j] + b[i]
        end
    end
    return c
end

plus_cols(a::AbstractArray, b::AbstractArray) = plus_cols!(similar(a), a, b)

plus_cols(scope::Scope, a::AbstractValue, b::AbstractValue) = Constant(plus_cols(a.data, b.data))

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    defn = :(function plus_cols(scope::GradScope, a::$A, b::$B)
        c = Variable(plus_cols(a.data, b.data))
        push_callback!(scope, ReversePlusCols(c, a, b))
        return c
    end)
    eval(defn)
end
