"""Finite difference gradient checking.

*Arguments*\\
g : Function that computes and stores the gradient of theta.\\
c : Cost function.\\
theta : The Component or Variable to run gradient check on.\\
eps : Step size for gradient approximation. (default: 1e-6)\\
tol : Tolerance. (default: 1e-6)\\
verbose : If true print information about success/failure. (default: true)\\
throwerr : If true throw an error when checks fail. (default: true)

*Returns*\\
passed : true if checks pass otherwise false.

*Throws*\\
ErrorException : If checks fail and throwerr is true.

*Methods*\\
gradcheck(g::Function, c::Function, theta::Component; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)\\
gradcheck(g::Function, c::Function, theta::Variable; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)

*Usage*
```
x, y = GradVariable(randn(4)), randn(4)
g() = gradient!(Cost.mse, x, y)
c() = Cost.mse(x, y)
check_gradients(g, c, x)
```
"""
function check_gradients end

function check_gradients(g, c, theta::Component; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)
    g()
    passed = true
    for (name, param) in getnamedparams(theta)
        for j = 1:size(param, 2)
            for i = 1:size(param, 1)
                xij = param.data[i,j]
                param.data[i,j] = xij + eps
                lp = c()
                param.data[i,j] = xij - eps
                lm = c()
                param.data[i,j] = xij
                dxij = (lp - lm) / (2 * eps)
                if abs(dxij - param.grad[i,j]) > tol
                    errmsg = "Finite difference gradient check failed!"
                    errelm = "  name => $name, index => ($i, $j), ratio => $(dxij / param.grad[i,j])"
                    errdsc = "  |$(dxij) - $(param.grad[i,j])| > $tol"
                    if throwerr
                        error("$errmsg\n$errelm\n$errdsc")
                    else
                        passed = false
                        if verbose
                            println("$errmsg\n$errelm\n$errdsc")
                        end
                    end
                end
            end
        end
    end
    if verbose
        status = passed ? "passed" : "failed"
        println("Finite difference gradient check $(status)!")
    end
    return passed
end

function check_gradients(g, c, param::Variable; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)
    g()
    passed = true
    for j = 1:size(param, 2)
        for i = 1:size(param, 1)
            xij = param.data[i,j]
            param.data[i,j] = xij + eps
            lp = c()
            param.data[i,j] = xij - eps
            lm = c()
            param.data[i,j] = xij
            dxij = (lp - lm) / (2 * eps)
            if abs(dxij - param.grad[i,j]) > tol
                errmsg = "Finite difference gradient check failed!"
                errelm = "  index => ($i, $j), ratio => $(dxij / param.grad[i,j])"
                errdsc = "  |$(dxij) - $(param.grad[i,j])| > $tol"
                if throwerr
                    error("$errmsg\n$errelm\n$errdsc")
                else
                    passed = false
                    if verbose
                        println("$errmsg\n$errelm\n$errdsc")
                    end
                end
            end
        end
    end
    if verbose
        status = passed ? "passed" : "failed"
        println("Finite difference gradient check $(status)!")
    end
    return passed
end
