
type ReverseDot{A<:AbstractValue,B<:AbstractValue} <: ReverseOperation
    c::Variable
    a::A
    b::B
end

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    updates = Any[]
    A <: Variable && push!(updates, :(a.grad[i,j] += c.grad[j] * b.data[i,j]))
    B <: Variable && push!(updates, :(b.grad[i,j] += c.grad[j] * a.data[i,j]))
    update_block = Expr(:block, updates...)
    defn = :(function call(rop::ReverseDot{$A,$B})
        c = rop.c
        a = rop.a
        b = rop.b
        @flimsy_inbounds for j = 1:size(a, 2), i = 1:size(a, 1)
            $update_block
        end
        nothing
    end)
    eval(defn)
end

function dot!(c::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    m, n = size(a)
    for j = 1:size(a, 2), i = 1:size(a, 1)
        c[1,j] += a[i,j] * b[i,j]
    end
    return c
end

function Base.dot(scope::Scope, a::AbstractValue, b::AbstractValue)
    asz = size(a)
    bsz = size(b)
    asz == bsz || throw(OperationError("no dot for sizes a: $asz, b: $bsz"))
    c_data = zeros(FloatX, 1, asz[2])
    return Constant(dot!(c_data, a.data, b.data))
end

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    defn = :(function Base.dot(scope::GradScope, a::$A, b::$B)
        # println("-----------------> dot{$A,$B}")
        asz = size(a)
        bsz = size(b)
        asz == bsz || throw(OperationError("no dot for sizes a: $asz, b: $bsz"))
        c_data = zeros(FloatX, 1, asz[2])
        c = Variable(dot!(c_data, a.data, b.data))
        push_callback!(scope, ReverseDot(c, a, b))
        return c
    end)
    eval(defn)
end

