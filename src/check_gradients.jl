"""Finite difference gradient checking.

*Arguments*\\
f : Cost function.\\
params : The item run gradient check on.\\
args : Extra arg to pass to the cost function.\\
eps : Step size for gradient approximation. (default: 1e-6)\\
tol : Tolerance. (default: 1e-6)\\
verbose : If true print information about success/failure. (default: true)\\
throwerr : If true throw an error when checks fail. (default: true)

*Returns*\\
passed : true if checks pass otherwise false.

*Throws*\\
ErrorException : If checks fail and throwerr is true.

*Usage*
```
using Flimsy
using Flimsy.Components
m, n = 4, 5
target = randn(m, n)
params = setup(ValueComponent(value=randn(m, n)); dynamic=true)
@component cost(params::ValueComponent, target) = Cost.mse(params.value, target)
check_gradients(cost, params, target)
```
"""
function check_gradients end

function check_gradients(f::Function, param::Variable, name; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)
    for j = 1:size(param, 2)
        for i = 1:size(param, 1)
            xij = param.data[i,j]
            param.data[i,j] = xij + eps
            lp = f()
            param.data[i,j] = xij - eps
            lm = f()
            param.data[i,j] = xij
            dxij = (lp - lm) / (2 * eps)
            if abs(dxij - param.grad[i,j]) > tol
                errmsg = "Finite difference gradient check failed!"
                errelm = "  name => $name, index => ($i, $j), ratio => $(dxij / param.grad[i,j])"
                errdsc = "  |$(dxij) - $(param.grad[i,j])| > $tol"
                if throwerr
                    error("$errmsg\n$errelm\n$errdsc")
                else
                    if verbose
                        println("$errmsg\n$errelm\n$errdsc")
                    end
                    return false
                end
            end
        end
    end
    return true
end

function check_gradients{V<:Variable}(f::Function, params::AbstractArray{V}, name; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)
    for param in params
        if !check_gradients(f, param, name; eps=eps, tol=tol, verbose=verbose, throwerr=throwerr)
            return false
        end
    end
    return true
end

function check_gradients(f::Function, params::Dict, name; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)
    for (name, value) in params
        if !check_gradients(f, value, name; eps=eps, tol=tol, verbose=verbose, throwerr=throwerr)
            return false
        end
    end
    return true
end

function check_gradients(f::Function, params::Runtime, args...; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)
    passed = true
    f(params, args...; grad=true)
    g = () -> f(params, args...; grad=false)

    for (name, value) in convert(Dict, params.component)
        if !check_gradients(g, value, name; eps=eps, tol=tol, verbose=verbose, throwerr=throwerr)
            return false
        end
    end
    if verbose
        status = passed ? "passed" : "failed"
        println("Finite difference gradient check $(status)!")
    end
    return passed
end