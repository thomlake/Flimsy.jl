
using Iterators

type ReverseAffine{W<:AbstractValue,X<:AbstractValue,B<:AbstractValue} <: ReverseOperation
    y::Variable
    w::W
    x::X
    b::B
end

for (W, X, B) in product([(Constant,Variable) for i = 1:3]...)
    any(T -> T <: Variable, [W, X, B]) || continue

    updates = Any[]
    W <: Variable && push!(updates, :(add_to_A_mul_Bt!(w.grad, y.grad, x.data)))
    X <: Variable && push!(updates, :(add_to_At_mul_B!(x.grad, w.data, y.grad)))
    B <: Variable && push!(updates, :(@flimsy_inbounds for j = 1:size(y, 2)
        for i = 1:size(y, 1)
            b.grad[i] += y.grad[i,j]
        end
    end))
    update_block = Expr(:block, updates...)
    defn = :(function call(rop::ReverseAffine{$W,$X,$B})
        y = rop.y
        w = rop.w
        x = rop.x
        b = rop.b
        $update_block
        nothing
    end)
    eval(defn)
end

function affine!(y::AbstractMatrix, w::AbstractMatrix, x::AbstractMatrix, b::AbstractMatrix)
    ysz = size(y)
    wsz = size(w)
    xsz = size(x)
    bsz = size(b)

    bsz == (wsz[1], 1) || throw(OperationError("b must be a $(wsz[1])x1 matrix"))
    ysz == (wsz[1], xsz[2]) || throw(OperationError("y must be a $(wsz[1])x$(xsz[2]) matrix"))

    A_mul_B!(y, w, x)
    @flimsy_inbounds for j = 1:xsz[2]
        for i = 1:wsz[1]
            y[i,j] += b[i]
        end
    end
    return y
end

affine(w::AbstractMatrix, x::AbstractMatrix, b::AbstractMatrix) = affine!(zeros(FloatX, size(w, 1), size(x, 2)), w, x, b)

affine(scope::Scope, w::AbstractValue, x::AbstractValue, b::AbstractValue) = Constant(affine(w.data, x.data, b.data))

for (W, X, B) in product([(Constant, Variable) for i = 1:3]...)
    any(T -> T <: Variable, [W, X, B]) || continue
    defn = :(function affine(scope::GradScope, w::$W, x::$X, b::$B)
        y = Variable(affine(w.data, x.data, b.data))
        push_callback!(scope, ReverseAffine(y, w, x, b))
        return y
    end)
    eval(defn)
end
