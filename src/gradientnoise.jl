
type GradientNoise{F<:AbstractFloat,I<:Integer}
    eta::F
    gamma::F
    timestep::I
end

GradientNoise(eta, gamma::AbstractFloat=0.55) = GradientNoise(eta, gamma, 1)

Base.step(noise::GradientNoise) = noise.timestep += 1

gradient_noise_scalar(noise::GradientNoise) = noise.eta / ((1 + noise.timestep)^noise.gamma)

function Base.call(noise::GradientNoise, theta::Component)
    sigma2 = gradient_noise_scalar(noise)
    for param in getparams(theta)
        n = sigma2 .* randn(size(param))
        for i in eachindex(param)
            param.grad[i] += n[i]
        end
    end
end
