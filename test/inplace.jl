using Flimsy

facts("inplace") do
    context("add_to_A_mul_Bt!") do
        for (m, n) in [(7, 1), (1, 5), (5, 4)]
            for k in [1, 10]
                for ivalue in [0.0, 3.3, -5.7]
                    w = randn(m, n)
                    x = randn(n, k)
                    y = w * x

                    dw1 = A_mul_Bt(y, x)
                    dw2 = zeros(size(w)) + ivalue
                    Flimsy.add_to_A_mul_Bt!(dw2, y, x)
                    @fact dw2 --> roughly(dw1 + ivalue)
                end
            end
        end
    end
    context("add_to_At_mul_B!") do
        for (m, n) in [(7, 1), (1, 5), (5, 4)]
            for k in [1, 10]
                for ivalue in [0.0, 3.3, -5.7]
                    w = randn(m, n)
                    x = randn(n, k)
                    y = w * x

                    dx1 = At_mul_B(w, y)
                    dx2 = zeros(size(x)) + ivalue
                    Flimsy.add_to_At_mul_B!(dx2, w, y)
                    @fact dx2 --> roughly(dx1 + ivalue)
                end
            end
        end
    end
end
