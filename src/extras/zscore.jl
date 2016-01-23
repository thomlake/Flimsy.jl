
zscore(x, mu, sigma, sigma_min::AbstractFloat=1e-6) = (x .- mu) ./ max(sigma, sigma_min)

function zscore(x::Matrix, sigma_min::AbstractFloat=1e-6)
    mu = mean(x, 2)
    sigma = std(x, 2)
    return zscore(x, mu, sigma, sigma_min)
end

zscore(xs::Vector{Matrix}, mu, sigma, sigma_min::AbstractFloat=1e-6) = map(x->zscore(x, mu, sigma, sigma_min), xs)

function zscore(xs::Vector{Matrix}, sigma_min::AbstractFloat=1e-6)
    d = size(xs[1], 1)
    n = 0

    mu = zeros(d)
    for x in xs
        mu .+= sum(x, 2)
        n += size(x, 2)
    end
    mu ./= n

    sigma = zeros(d)
    for x in xs
        for j = 1:size(x, 2)
            for i = 1:d
                diff = (x[i,j] - mu[i])
                sigma[i] += diff * diff
            end
        end
    end
    sigma = sqrt(sigma ./ (n - 1))
    return zscore(xs, mu, sigma, 1e-6)
end
