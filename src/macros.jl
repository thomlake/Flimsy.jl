"""
Array of function names that Nimble should not
try and backpropage through
"""
NIMBLE_BLACKLIST_SYMBOLS = [
    :Var,
    :endof,
    :eachindex,
    :enumerate,
    :zip,
    :length,
    :println,
    :typeof,
    :push!,
    :append!,
    :Array,
    :-,
    :reverse,
]

"""
Predicate functions to match expressions Nimble
should ignore and not try and backpropage through.
Functions should return Bool.
"""
NIMBLE_BLACKLIST_PATTERNS = Function[]

isblacklisted(sym::Symbol) = sym in NIMBLE_BLACKLIST_SYMBOLS

function isblacklisted(expr::Expr)
    for f in NIMBLE_BLACKLIST_PATTERNS
        try
            result = f(expr)
            if result
                return true
            end
        end
    end
    return false
end

"""
Insert a BPStack as the first argument in a function signature.

*Input:*
    f(x::Int, y::Var)

*Output:*
    f(__nimble_bpstack__::BPStack, x::Int, y::Var)
"""
function backprop_signature(signature)
    new_signature = deepcopy(signature)
    insert!(new_signature.args, 2, Expr(:(::), :__nimble_bpstack__, :BPStack))
    return new_signature
end

"""
Rewrite `signature` so that variables with type `Var` are untyped
and create statments to convert corresponding arg to type `Var`.

*Input:*

    f(x::Int, y::Var)

*Output:*

    (f(x::Int, y), [y=Var(y)])
"""
function untyped_signature_and_conversions(signature)
    new_signature = deepcopy(signature)
    convert_stmts = Any[]
    for (i, arg) in enumerate(new_signature.args[1:end])
        if typeof(arg) <: Expr && length(arg.args) > 1 && arg.args[2] == :Var
            new_signature.args[i] = arg.args[1]
            x = Expr(:(=), arg.args[1], Expr(:call, :Var, arg.args[1]))
            push!(convert_stmts, x)
        end
    end
    return new_signature, convert_stmts
end

"""
Convert a function signature into a call of that function with
arguments of the same name.

*Input:*

    f(x::Int, y, z::Var)

*Output:*

    f(x, y, z)
"""
function signature_as_call(signature::Expr)
    callexpr = Expr(:call, signature.args[1])
    for arg in signature.args[2:end]
        push!(callexpr.args, typeof(arg) <: Expr ? arg.args[1] : arg)
    end
    return callexpr
end

"""
Rewrite function body so all non-blacklisted functions have a
BPStack instance as their first argument, i.e., `foo(a, b, c)`
becomes `foo(__nimble_bpstack__, a, b, c)`.
"""
function backprop_body(expr::Expr)
    head = expr.head
    args = deepcopy(expr.args)
    newargs = Any[]
    if head == :call
        if !isblacklisted(expr.args[1])
            push!(newargs, shift!(args))
            push!(newargs, Expr(:(::), :__nimble_bpstack__, BPStack))
        end
    end
    for arg in args
        try
            push!(newargs, backprop_body(arg))
        catch
            push!(newargs, arg)
        end
    end
    return Expr(expr.head, newargs...)
end

function component_parser(f::Expr)
    f.head == :(=) || f.head == :function || error("[@Nimble.component] expected = or function but got ", f.head)
    ok = length(f.args) == 2 && f.args[1].head == :call && f.args[2].head == :block
    ok || error("[@Nimble.component] expected (:call, :block) but got ", map(x->x.head, f.args))

    sig = f.args[1]
    body = f.args[2]

    bpsig = backprop_signature(sig)
    bpbody = backprop_body(body)

    usig, uconvert_stmts = untyped_signature_and_conversions(sig)
    wrapped_call = signature_as_call(sig)

    ubpsig, ubpconvert_stmts = untyped_signature_and_conversions(bpsig)
    wrapped_bpcall = signature_as_call(bpsig)

    fbp = Expr(f.head, bpsig, bpbody)
    fu = Expr(f.head, usig, Expr(:block, uconvert_stmts..., wrapped_call))
    fubp = Expr(f.head, ubpsig, Expr(:block, ubpconvert_stmts..., wrapped_bpcall))

    retblock = Any[f, fbp]
    if length(uconvert_stmts) > 0
        push!(retblock, fu)
    end
    if length(ubpconvert_stmts) > 0
        push!(retblock, fubp)
    end

    return Expr(:block, retblock...)
end

macro component(x::Expr)
    y = component_parser(x)
    return esc(y)
end
