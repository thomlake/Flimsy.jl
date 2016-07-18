
type ReverseLinear{W<:AbstractValue,X<:AbstractValue} <: ReverseOperation
    y::Variable
    w::W
    x::X
end

for (W, X) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    updates = Any[]
    W <: Variable && push!(updates, :(add_to_A_mul_Bt!(w.grad, y.grad, x.data)))
    X <: Variable && push!(updates, :(add_to_At_mul_B!(x.grad, w.data, y.grad)))
    update_block = Expr(:block, updates...)
    defn = :(function call(rop::ReverseLinear{$W,$X})
        y = rop.y
        w = rop.w
        x = rop.x
        $update_block
        nothing
    end)
    eval(defn)
end

linear!(y::AbstractArray, w::AbstractArray, x::AbstractArray) = A_mul_B!(y, w, x)

linear(w::AbstractArray, x::AbstractArray) = w * x

linear(scope::Scope, w::AbstractValue, x::AbstractValue) = Constant(linear(w.data, x.data))

for (W, X) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    defn = :(function linear(scope::GradScope, w::$W, x::$X)
        y = Variable(linear(w.data, x.data))
        push_callback!(scope, ReverseLinear(y, w, x))
        return y
    end)
    eval(defn)
end
