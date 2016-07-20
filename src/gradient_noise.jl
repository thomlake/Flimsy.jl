
type GradientNoise
    paramvec::Vector{Variable}
    eta::Float64
    gamma::Float64
    timestep::Int
end

GradientNoise(params, eta, gamma::AbstractFloat=0.55) = GradientNoise(convert(Vector, params), eta, gamma, 1)

Base.step(noise::GradientNoise) = noise.timestep += 1

gradient_noise_scalar(noise::GradientNoise) = noise.eta / ((1 + noise.timestep)^noise.gamma)

function call(noise::GradientNoise)
    sigma2 = gradient_noise_scalar(noise)
    for param in noise.paramvec
        nz = sigma2 .* randn(size(param))
        for i in eachindex(param)
            param.grad[i] += nz[i]
        end
    end
end
