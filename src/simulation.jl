!cpu && using CUDA

function NNp(n)
    n%L+1
end

function NNm(n)
    (n+L-2)%L+1
end

#=
  Sublattice period for the race-free parallel update with the 1/2 (∇²φ)² term.
  The ΔH for site x reads all sites within L1 distance ≤ 2.  To avoid races,
  same-sublattice sites must be at torus-L1 distance ≥ 3.
  We use a product-p coloring: site x → (x1-1)%p + p*((x2-1)%p) + p²*((x3-1)%p).
  For this to respect the torus (period p must divide L), p_sub is the smallest
  divisor of L that is ≥ 3 and satisfies L ≥ 2*p (so that min torus distance = p ≥ 3).
=#
const p_sub = let
    p = 3
    while p <= L
        if L % p == 0 && L >= 2*p
            break
        end
        p += 1
    end
    p > L && error("L=$L has no valid sublattice period (need a divisor p≥3 of L with L≥2p; try L divisible by 3 or 4)")
    p
end
const N_sub = p_sub^3  # number of sublattices per dissipative step

function ΔH(ϕ, m², x, q)
    ϕold = ϕ[x...]
    ϕt = ϕold + q
    Δϕ = ϕt - ϕold
    Δϕ² = ϕt^2 - ϕold^2

    ∑nn = (ϕ[NNp(x[1]), x[2], x[3]] + ϕ[x[1], NNp(x[2]), x[3]] + ϕ[x[1], x[2], NNp(x[3])]
         + ϕ[NNm(x[1]), x[2], x[3]] + ϕ[x[1], NNm(x[2]), x[3]] + ϕ[x[1], x[2], NNm(x[3])])

    # Axial next-nearest neighbours (±2 along each axis)
    ∑nnn_axial = (ϕ[NNp(NNp(x[1])), x[2], x[3]] + ϕ[NNm(NNm(x[1])), x[2], x[3]]
               + ϕ[x[1], NNp(NNp(x[2])), x[3]] + ϕ[x[1], NNm(NNm(x[2])), x[3]]
               + ϕ[x[1], x[2], NNp(NNp(x[3]))] + ϕ[x[1], x[2], NNm(NNm(x[3]))])

    # Diagonal next-nearest neighbours (±1 along two different axes)
    ∑nnn_diag = (ϕ[NNp(x[1]), NNp(x[2]), x[3]] + ϕ[NNp(x[1]), NNm(x[2]), x[3]]
              + ϕ[NNm(x[1]), NNp(x[2]), x[3]] + ϕ[NNm(x[1]), NNm(x[2]), x[3]]
              + ϕ[NNp(x[1]), x[2], NNp(x[3])] + ϕ[NNp(x[1]), x[2], NNm(x[3])]
              + ϕ[NNm(x[1]), x[2], NNp(x[3])] + ϕ[NNm(x[1]), x[2], NNm(x[3])]
              + ϕ[x[1], NNp(x[2]), NNp(x[3])] + ϕ[x[1], NNp(x[2]), NNm(x[3])]
              + ϕ[x[1], NNm(x[2]), NNp(x[3])] + ϕ[x[1], NNm(x[2]), NNm(x[3])])

    # Z * conventional kinetic term: Z/2 (∇φ)²
    ΔH_kin = Z * (3Δϕ² - Δϕ * ∑nn)

    # Higher-order kinetic term: 1/2 (∇²φ)²
    # From ΔH = q*{-6 L_x + (2d²+d)*q + Σ_{y~x} L_y} (d=3 → 2d²+d=21), where L_x = ∇²φ(x).
    # Σ_{y~x} L_y = ∑nnn_axial + 2d*φ_old + 2∑nnn_diag - 2d*∑nn  (d=3: coefficients 6 and 6)
    # Collecting: ΔH_lap = Δϕ*(21Δϕ + (2*6*3)*φ_old - 2*6*∑nn + ∑nnn_axial + 2∑nnn_diag)
    #                     = Δϕ*(21Δϕ + 42φ_old - 12∑nn + ∑nnn_axial + 2∑nnn_diag)
    ΔH_lap = Δϕ * (21Δϕ - 12∑nn + 42ϕold + ∑nnn_axial + 2∑nnn_diag)

    return ΔH_kin + ΔH_lap + 0.5m² * Δϕ² + 0.25λ * (ϕt^4 - ϕold^4)
end

function step(ϕ, m², x1, x2, x3)
    x = (x1, x2, x3)

    norm = cos(2pi*rand())*sqrt(-2*log(rand()))
    q = Rate * norm

    δH = ΔH(ϕ, m², x, q)

    ϕ[x...] += q * (rand() < exp(-δH/T))
end

##
if cpu

function sweep(ϕ, m², n)
    a = n % p_sub
    b = (n ÷ p_sub) % p_sub
    c = n ÷ p_sub^2
    M = (L - 1) ÷ p_sub + 1
    Threads.@threads for l in 0:M^3-1
        i1 = l % M
        i2 = (l ÷ M) % M
        i3 = l ÷ M^2
        x1 = a + p_sub*i1 + 1
        x2 = b + p_sub*i2 + 1
        x3 = c + p_sub*i3 + 1
        if x1 <= L && x2 <= L && x3 <= L
            step(ϕ, m², x1, x2, x3)
        end
    end
end

else

function _sweep(ϕ, m², n)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = gridDim().x * blockDim().x
    a = n % p_sub
    b = (n ÷ p_sub) % p_sub
    c = n ÷ p_sub^2
    M = (L - 1) ÷ p_sub + 1

    for l in index:stride:M^3-1
        i1 = l % M
        i2 = (l ÷ M) % M
        i3 = l ÷ M^2
        x1 = a + p_sub*i1 + 1
        x2 = b + p_sub*i2 + 1
        x3 = c + p_sub*i3 + 1
        if x1 <= L && x2 <= L && x3 <= L
            step(ϕ, m², x1, x2, x3)
        end
    end
end

_sweep_gpu = @cuda launch=false _sweep(ArrayType{FloatType}(undef,(L,L,L)), zero(FloatType), 0)

const N = ((L-1)÷p_sub+1)^3
config = launch_configuration(_sweep_gpu.fun)
const threads = min(N, config.threads)
const blocks = cld(N, threads)

sweep = (ϕ, m², n) -> _sweep_gpu(ϕ, m², n; threads=threads, blocks=blocks)

end

function dissipative(ϕ, m²)
    for n in 0:N_sub-1
        sweep(ϕ, m², n)
    end
end

function thermalize(ϕ, m², N)
    for _ in 1:N
        dissipative(ϕ, m²)
    end
end

##
if cpu

function calc_total_energy(ϕ, m², Z)
    H = 0.0
    for x3 in 1:L, x2 in 1:L, x1 in 1:L
        ϕ0 = ϕ[x1, x2, x3]

        ϕp_x = ϕ[NNp(x1), x2, x3]
        ϕm_x = ϕ[NNm(x1), x2, x3]
        ϕp_y = ϕ[x1, NNp(x2), x3]
        ϕm_y = ϕ[x1, NNm(x2), x3]
        ϕp_z = ϕ[x1, x2, NNp(x3)]
        ϕm_z = ϕ[x1, x2, NNm(x3)]

        lapl  = (ϕp_x + ϕm_x - 2ϕ0) + (ϕp_y + ϕm_y - 2ϕ0) + (ϕp_z + ϕm_z - 2ϕ0)
        grad2 = (ϕp_x - ϕ0)^2 + (ϕp_y - ϕ0)^2 + (ϕp_z - ϕ0)^2

        H += 0.5 * lapl^2 + (Z / 2.0) * grad2 + (m² / 2.0) * ϕ0^2 + (λ / 4.0) * ϕ0^4
    end
    return H
end

else

function _energy_kernel!(H_arr, ϕ, m², Z)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= L^3
        flat = idx - 1
        x1 = flat % L + 1
        x2 = (flat ÷ L) % L + 1
        x3 = flat ÷ L^2 + 1

        ϕ0 = ϕ[x1, x2, x3]

        ϕp_x = ϕ[NNp(x1), x2, x3]
        ϕm_x = ϕ[NNm(x1), x2, x3]
        ϕp_y = ϕ[x1, NNp(x2), x3]
        ϕm_y = ϕ[x1, NNm(x2), x3]
        ϕp_z = ϕ[x1, x2, NNp(x3)]
        ϕm_z = ϕ[x1, x2, NNm(x3)]

        lapl  = (ϕp_x + ϕm_x - 2ϕ0) + (ϕp_y + ϕm_y - 2ϕ0) + (ϕp_z + ϕm_z - 2ϕ0)
        grad2 = (ϕp_x - ϕ0)^2 + (ϕp_y - ϕ0)^2 + (ϕp_z - ϕ0)^2

        H_arr[idx] = FloatType(0.5) * lapl^2 + (Z * FloatType(0.5)) * grad2 +
                     (m² * FloatType(0.5)) * ϕ0^2 + (λ * FloatType(0.25)) * ϕ0^4
    end
    return nothing
end

function calc_total_energy(ϕ, m², Z)
    Ntot = L^3
    H_arr = CuArray{FloatType}(undef, Ntot)
    en_threads = 256
    en_blocks = cld(Ntot, en_threads)
    @cuda threads=en_threads blocks=en_blocks _energy_kernel!(H_arr, ϕ, m², Z)
    return sum(H_arr)
end

end
