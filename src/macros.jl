"""
Array of function names that Flimsy should not
try and backpropage through
"""
BLACKLIST_SYMBOLS = [
    :Variable,
    :endof,
    :eachindex,
    :enumerate,
    :zip,
    :length,
    :println,
    :size,
    :eltype,
    :typeof,
    :push!,
    :append!,
    :Array,
    :-,
    :+,
    :reverse,
    :map,
]

"""
Predicate functions to match expressions Flimsy
should ignore and not try and backpropage through.
Functions should return Bool.
"""
BLACKLIST_PATTERNS = Function[]

isblacklisted(sym::Symbol) = sym in BLACKLIST_SYMBOLS

function isblacklisted(expr::Expr)
    for f in BLACKLIST_PATTERNS
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
    f(x::Int, y::Variable)

*Output:*
    f(__flimsy_bpstack__::BPStack, x::Int, y::Variable)
"""
function backprop_signature(signature)
    new_signature = deepcopy(signature)
    insert!(new_signature.args, 2, Expr(:(::), :__flimsy_bpstack__, :BPStack))
    return new_signature
end

"""
Rewrite `signature` so that variables with type `Variable` are untyped
and create statments to convert corresponding arg to type `Variable`.

*Input:*

    f(x::Int, y::Variable)

*Output:*

    (f(x::Int, y), [y=Variable(y)])
"""
function untyped_signature_and_conversions(signature)
    new_signature = deepcopy(signature)
    convert_stmts = Any[]
    for (i, arg) in enumerate(new_signature.args[1:end])
        if typeof(arg) <: Expr && length(arg.args) > 1 && arg.args[2] == :Variable
            new_signature.args[i] = arg.args[1]
            x = Expr(:(=), arg.args[1], Expr(:call, :Variable, arg.args[1]))
            push!(convert_stmts, x)
        end
    end
    return new_signature, convert_stmts
end

"""
Convert a function signature into a call of that function with
arguments of the same name.

*Input:*

    f(x::Int, y, z::Variable)

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
becomes `foo(__flimsy_bpstack__, a, b, c)`.
"""
function backprop_body(expr::Expr)
    head = expr.head
    args = deepcopy(expr.args)
    newargs = Any[]
    if head == :call
        if !isblacklisted(expr.args[1])
            push!(newargs, shift!(args))
            push!(newargs, Expr(:(::), :__flimsy_bpstack__, BPStack))
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

function flimsy_parser(f::Expr)
    f.head == :(=) || f.head == :function || error("[@flimsy] expected = or function but got ", f.head)
    ok = length(f.args) == 2 && f.args[1].head == :call && f.args[2].head == :block
    ok || error("[@flimsy] expected (:call, :block) but got ", map(x->x.head, f.args))

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

macro flimsy(x::Expr)
    y = flimsy_parser(x)
    return esc(y)
end
