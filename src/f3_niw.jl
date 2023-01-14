"""
    NormalInvWishart(b, Ω, Ψ, d)

Object with parameters for a Normal-Inverse-Wishart distribution:

    Σ ∼ 𝐼𝑊(Ψ; d)
    β|Σ ∼ 𝑁(b, Σ ⊗ Ω)
"""
struct NormalInvWishart
    b::Vector{Float64}
    Ω::Matrix{Float64}
    Ψ::Matrix{Float64}
    d::Float64
end

"""
    N = setniw(b, Ω, Ψ, d)

Build a `NormalInvWishart` distribution object.
"""
setniw(b, Ω, Ψ, d) = NormalInvWishart(vect(b), mat.([Ω, Ψ])..., float(d))

"""
    R, N = size(niw::NormalInvWishart)

### Output

- `R::Int64`: length of `β`

- `N::Int64`: size of `Σ`
"""
function size(niw::NormalInvWishart)
    R = length(niw.b)
    N = size(niw.Ψ, 1)
    return R, N
end
size(niw::NormalInvWishart, i::Int64) = size(niw)[i]

"""
    β, Σ = mean(niw::NormalInvWishart)
"""
function mean(niw::NormalInvWishart)
    Σ = niw.Ψ / (niw.d - size(niw.Ψ, 1) - 1)
    β = niw.b
    return β, Σ
end

"""
    β, Σ = mode(niw::NormalInvWishart)
"""
function mode(niw::NormalInvWishart)
    Σ = niw.Ψ / (niw.d + size(niw.Ψ, 1) + 1)
    β = niw.b
    return β, Σ
end


"""
    β, Σ = rand(niw::NormalInvWishart)

Draw a random element from the Normal-Inverse-Wishart distribution parameterized by `niw`.
"""
function rand(niw::NormalInvWishart)

    # draw Σ
    sigmadist = InverseWishart(niw.d, niw.Ψ)
    Σ = rand(sigmadist)

    # draw β
    betadist = MvNormal(niw.b, kron(Σ, niw.Ω))
    β = rand(betadist)

    return β, Σ
end

function logpdf(β::Vector{Float64}, Σ::Matrix{Float64}, niw::NormalInvWishart)

    # draw Σ
    sigmadist = InverseWishart(niw.d, niw.Ψ)
    sigmaterm = Distributions.logpdf(sigmadist, Σ)

    # draw β
    betadist = MvNormal(niw.b, kron(Σ, niw.Ω))
    betaterm = Distributions.logpdf(betadist, β)

    density = sigmaterm + betaterm
    return density
end



# ------------------------------------------------------------------
# BAYESIAN MULTIVARIATE LINEAR REGRESSION
# ------------------------------------------------------------------

function logmvgamma(a, N)
    logΓ = loggamma(a)
    for n in 2:N
        logΓ += ((n - 1) / 2) * log(pi) + loggamma(a + (1 - n) / 2)
    end
    return logΓ
end

"""
    posterior = bayesreg(y, x, prior; intercept::Bool=true)

Compute the posterior distribution of the Bayesian multivariate linear regression. The model is

        yₜ = B xₜ + ϵₜ
        ϵₜ ∼ 𝑁(0, Σ)

The conjugate `prior` is a Normal-Inverse-Wishart distribution.

        Σ    ∼ 𝐼𝑊(Ψ; d)
        β|Σ  ∼ 𝑁(b, Σ ⊗ Ω)
        
where `β = vec(B')`. `Ω` captures prior covariance between coefficients of the same equation.

The `posterior` is also a Normal-Inverse-Wishart distribution.
"""
function bayesreg(y::VecOrMat{U},
    x::VecOrMat{J},
    prior::NormalInvWishart;
    intercept::Bool=true) where {U<:Real,J<:Real}

    # ensure sizes are consistent
    nlh = size(y, 2)
    T, nind = size(x)
    nrh = nind + intercept
    Rprior, Nprior = size(prior)
    @assert Rprior == nlh * nrh
    @assert Nprior == nlh

    b0 = prior.b
    Ω0 = prior.Ω
    Ψ0 = prior.Ψ
    d0 = prior.d
    B0 = reshape(b0, nrh, nlh)

    # prior: 
    #       Σ ∼ 𝐼𝑊(Ψ0, d0)
    #       β|Σ ∼ 𝑁(b0, Σ ⊗ Ω0)

    # add intercept to x, if necessary
    if intercept
        X = [ones(T) x]
    else
        X = x
    end

    # Normal component N(b, Σ ⊗ Ω)
    # invΩ0 = diagm(0 => 1 ./ diag(Ω0))
    invΩ0 = inv(Ω0)
    Ω = inv(X' * X + invΩ0) |> makehermitian
    B = Ω * (X' * y + invΩ0 * B0)
    b = B[:]

    # Inverse-Wishart Component: 𝐼𝑊(Ψ, d)
    e = y .- X * B
    ee = e' * e
    eqparcov = (B .- B0)' * invΩ0 * (B .- B0)
    d = d0 + T
    Ψ = Ψ0 .+ ee + eqparcov |> makehermitian

    # posterior distribution
    posterior = setniw(b, Ω, Ψ, d)

    # calculate marginal
    # DΩ = sqrt.(Ω0)
    # DΨ = sqrt.(inv(Ψ0))
    DΩ = Matrix(cholesky(Ω0).L)
    DΨ = Matrix(cholesky(inv(Ψ0)).L)
    # aux1 = LinearAlgebra.Symmetric(DΩ' * X' * X * DΩ)
    # aux2 = LinearAlgebra.Symmetric(DΨ' * (ee + eqparcov) * DΨ)
    aux1 = DΩ' * X' * X * DΩ
    aux2 = DΨ' * (ee + eqparcov) * DΨ
    eig1 = eigen(aux1).values
    eig2 = eigen(aux2).values
    marginal = -(nlh * T / 2) * log(pi)
    marginal += logmvgamma((T + d0) / 2, nlh) - logmvgamma(d0 / 2, nlh)
    marginal += -(T / 2) * log(det(Ψ0))
    marginal += -(nlh / 2) * sum(log.(1 .+ eig1))
    marginal += -((T + d0) / 2) * sum(log.(1 .+ eig2))

    #= inefficient calculation of marginal
    marg2 = -(nlh * T / 2) * log(pi)
    marg2 += logmvgamma((T + d0) / 2, nlh) - logmvgamma(d0 / 2, nlh)
    marg2 += -(nlh / 2) * log(det(Ω0))
    marg2 += (d0 / 2) * log(det(Ψ0))
    marg2 += -(nlh / 2) * log(det(X' * X + inv(Ω0)))
    marg2 += -((T + d0) / 2) * log(det(Ψ))
    =#

    return marginal, posterior
end


