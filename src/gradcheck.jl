function gradcheck(g::Function, c::Function, theta::Component; eps::AbstractFloat=1e-6, tol::AbstractFloat=1e-6, verbose::Bool=true, throwerr::Bool=true)
    g()
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
                        println("$errmsg\n$errelm\n$errdsc")
                    end
                end
            end
        end
    end
    if verbose
        println("gradcheck passed")
    end
    return true
end
