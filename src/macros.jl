
type FlimsyParseError <: Exception
    msg::ASCIIString
end

Base.showerror(io::IO, e::FlimsyParseError) = print(io, "FlimsyParseError: ", e.msg)


"""
Array of function names that Flimsy should 
not backpropage through
"""
const DEFAULT_BLACKLIST = [
    :Variable,
    :Array,
    :size,
    :length,
    :eachindex,
    :endof,
    :reverse,
    :enumerate,
    :zip,
    :println,
    :eltype,
    :typeof,
    :push!,
    :append!,
    :!,
    :rand,
]

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
Recursively rewrite expr so all non-blacklisted function 
calls have a BPStack instance as their first argument.

For example
    foo(a, b, c)`
is transformed to
    foo(__flimsy_bpstack__, a, b, c)
"""
function rewrite_for_backprop(expr::Expr, blacklist::Vector{Symbol})
    
    head = expr.head
    args = expr.args
    newargs = Any[]

    # Special cases: line numbers and directives
    if head == :line
        return expr, blacklist
    elseif head == :macrocall && expr.args[1] == symbol("@blacklist")
        return nothing, Symbol[blacklist..., expr.args[2:end]...]
    end
    
    if head == :call
        if !(args[1] in blacklist)
            push!(newargs, args[1])
            push!(newargs, :__flimsy_bpstack__)
        end
    elseif head == :block
    elseif head == :tuple
    elseif head == :dict
    elseif head == :vect
    elseif head == :(=)
    elseif head == :(:)
    elseif head == :(=>)
    elseif head == :(.)
    elseif head == :quote
    elseif head == :ref
    elseif head == :comparison
    elseif head == :return
    elseif head == :for
    elseif head == :while
    elseif head == :comprehension
    elseif head == :typed_comprehension
    elseif head == :call
    else
        throw(FlimsyParseError("Unsupported Expr: ($head, $args)"))
    end
    
    inner_blacklist = blacklist

    for arg in args
        if typeof(arg) <: Expr
            newarg, inner_blacklist = rewrite_for_backprop(arg, inner_blacklist)
            if newarg != nothing
                push!(newargs, newarg)
            end
        else
            push!(newargs, arg)
        end
    end
    return Expr(head, newargs...), blacklist
end

function flimsy_parse(f::Expr)
    f.head == :(=) || f.head == :function || throw(FlimsyParseError("expected = or function but got $(f.head)"))
    ok = length(f.args) == 2 && f.args[1].head == :call && f.args[2].head == :block
    ok || throw(FlimsyParseError("expected :call or :block but got $(map(x->x.head, f.args))"))

    sig = f.args[1]
    body = f.args[2]

    bpsig = backprop_signature(sig)
    bpbody = rewrite_for_backprop(body, DEFAULT_BLACKLIST)[1]

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
    y = flimsy_parse(x)
    return esc(y)
end
