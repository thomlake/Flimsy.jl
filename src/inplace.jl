
function add_to_A_mul_Bt!{T<:AbstractFloat}(c::Matrix{T}, b::Matrix{T}, a::Matrix{T})
    @inbounds for k = 1:size(a, 2)
        for j = 1:size(c, 2)
            @simd for i = 1:size(c, 1)
                c[i,j] += b[i,k] * a[j,k]
            end
        end
    end
end

function add_to_At_mul_B!{T<:AbstractFloat}(c::Matrix{T}, b::Matrix{T}, a::Matrix{T})
    @inbounds for j = 1:size(a, 2)
        for i = 1:size(b, 2)
            @simd for k = 1:size(a, 1)
                c[i,j] += b[k,i] * a[k,j]
            end
        end
    end
end