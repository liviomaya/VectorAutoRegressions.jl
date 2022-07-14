########################################################################
# EM algorithm based on Shumay & Stoffer (1982)
########################################################################

function estep(X::VecOrMat{Float64}, v::StackedVAR, fp::VecOrMat{<:Real},
    μ0::Vector{Float64}, Σ0::VecOrMat{Float64})

    Z, W, Z0, W0, Wlag, lhd = kalmansmoother(X, v, fp, μ0, Σ0, getWlag=true)
    return Z, W, Z0, W0, Wlag, lhd
end

function mstep(X::VecOrMat{Float64},
    fp::VecOrMat{<:Real},
    Z::VecOrMat{Float64},
    W::Array{Float64,3},
    Z0::Vector{Float64},
    W0::Matrix{Float64},
    Wlag::Array{Float64,3};
    intercept::Bool=true)

    T, N = size(X)
    R = size(W, 2)

    # pre-allocate output matrices
    A = zeros(R + intercept, R + intercept)
    B = zeros(R, R + intercept)
    C = zeros(R, R)

    # iterate and compute
    for t in 1:T
        fpc = fp[t, :]
        Zc = Z[t, :] .- fpc
        Wc = W[t, :, :]
        Zp = (t == 1) ? Z0 : Z[t-1, :]
        Wp = (t == 1) ? W0 : W[t-1, :, :]
        Wlagc = Wlag[t, :, :]

        if intercept
            Aterm = [1 Zp'; Zp (Zp*Zp'.+Wp)]
            Bterm = [Zc (Zc * Zp' .+ Wlagc)]
        else
            Aterm = Zp * Zp' .+ Wp
            Bterm = Zc * Zp' .+ Wlagc
        end

        A .+= Aterm
        B .+= Bterm
        C .+= (Zc * Zc') .+ Wc
    end

    # define next parameter vector
    Ai = inv(A)
    Φst = B * Ai

    nμ = intercept ? Φst[:, 1] : zeros(R)
    nΦ = intercept ? Φst[:, 2:end] : Φst
    nΣ = (C .- B * Ai * B') / T
    nμ0 = Z0

    return nμ, nΦ, nΣ, nμ0
end

# TODO: Change notation: replace "data" for "X" ets

"""
    v, Z, W, L = emalgo(X; <kwargs>)

Run the EM algorithm as in Shumway and Stoffer (1982).

### Output 

- `v::VAR`: fitted VAR process.

- `Z::Matrix{Float64}`: Expected value of the distribution of states conditional on all data.

- `W::Matrix{Float64}`: Covariance matrices of the distributions conditional on all data.

- `L::Float64`: Log-Likelihood.

### Arguments

- `X::VecOrMat{<:Real}`: array with sample to be fitted (observations in rows, variables in columns).

### Keyword Arguments

- `P::Int64=1`: number of lags in the VAR.

- `intercept::Bool=true`: whether the VAR model to be fitted contains a constant term.

- `Σ0::Matrix{Float64}`: covariance matrix of the probability distribution of the initial condition. Defaults to identity matrix of size `N × P`, where `N` is the length of each data point (that is, `size(X,2)`).

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `data`  correspond to the same period.

- `itermax::Int64`: Maximum number of EM iterations.

- `tol::Float64`: tolerance for convergence in log-likelihood.

- `report::Bool`: Set `true` to display distance, number of step and log-likelihood as the algorithm runs.
"""
function emalgo(X::VecOrMat{Float64};
    P::Int64=1,
    intercept::Bool=true,
    Σ0::Matrix{Float64}=diagm(0 => ones(P * size(X, 2))),
    fp::VecOrMat{U}=zeros(size(X)),
    itermax::Int64=1000,
    tol::Float64=1e-4,
    report::Bool=true) where {U<:Real}

    # sizes
    N = size(X, 2)
    R = N * P
    @assert R == size(Σ0, 1)

    # Parameter guess
    μG = zeros(N)
    ΨG = [zeros(N, N) for _ in 1:P]
    ΣG = collect(I(N))
    μ0 = zeros(R)
    v = setvar(μG, ΨG, ΣG)
    lhd = -1e10

    # iteration
    fpstack = stackfp(fp, v)
    report && (progress = ProgressThresh(tol, "EM Iteration: "))
    for iter in 1:itermax
        # stack parameters
        V = stack(v)

        # EM steps
        Z, W, Z0, W0, Wlag, Tlhd = estep(X, V, fpstack, μ0, Σ0)
        nμ, nΦ, nΣ, nμ0 = mstep(X, fpstack, Z, W, Z0, W0, Wlag,
            intercept=intercept)

        # build new VAR
        nΨunstack = [nΦ[1:N, (p-1)*N.+(1:N)] for p in 1:P]
        Tv = setvar(nμ[1:N], nΨunstack, nΣ[1:N, 1:N])
        Tμ0 = nμ0

        # re-calculate distance
        distance = norm(lhd - Tlhd)
        (distance < tol) && break
        report && ProgressMeter.update!(progress, distance,
            showvalues=[(:Iteration, iter), (:LogLikelihood, Tlhd)])

        # update 
        v = deepcopy(Tv)
        μ0 = Tμ0
        lhd = Tlhd

        # report if convergence failed
        if (iter == itermax)
            println("")
            println("EM algorithm: convergence failed.")
        end
    end
    println("")
    println("")

    # build result with unstacked kalman smoother
    Z, W = kalmansmoother(X, v, fp=fp, μ0=μ0, Σ0=Σ0)

    return v, Z, W, lhd
end


# --------------------------------------------------------------------
# OLS ESTIMATION and EM ALGORITHM
# --------------------------------------------------------------------

function regdata(data::VecOrMat{<:Real};
    P::Int64=1,
    intercept::Bool=true)

    T, N = size(data)
    y = data[P+1:end, :]
    x = ones(T - P, P * N + intercept)
    for lag in 1:P
        columns = (lag - 1) * N .+ (1:N) .+ intercept
        x[:, columns] = data[P-lag+1:end-lag, :]
    end

    return y, x
end

"""
    v = fitols(data; <kwargs>)

Fit VAR model to the data.

### Output 

- `v::VAR`: fitted VAR process.

### Arguments

- `data::VecOrMat{<:Real}`: array with sample to be fitted (observations in rows, variables in columns).

### Keyword Arguments

- `P::Int64=1`: number of lags in the VAR.

- `intercept::Bool=true`: whether the VAR model to be fitted contains a constant term.

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `data`  correspond to the same period.

"""
function fitols(data::Matrix{<:Real};
    P::Int64=1,
    intercept::Bool=true,
    fp::VecOrMat{R}=zeros(size(data))) where {R<:Real}

    T, N = size(data)
    y, x = regdata(data, P=P, intercept=intercept)

    b = inv(x' * x) * x' * (y .- fp[P+1:T, :])
    yfit = x * b .+ fp[P+1:T, :]
    e = y .- yfit
    Σ = cov(e)

    μ = intercept ? b[1, :] : zeros(N)
    getmat(p) = b[intercept.+(1+N*(p-1):N*p), :]' |> Matrix
    Ψ = getmat.(1:P)
    v = setvar(μ, Ψ, Σ)

    return v
end

fitols(data::Vector{<:Real}; P::Int64=1, intercept::Bool=true,
    fp::VecOrMat{R}=zeros(size(data))) where {R<:Real} =
    fitols(mat(data), P=P, intercept=intercept, fp=fp)

function eminitialize(data::VecOrMat{<:Real})
    datainitial = copy(data)
    for index in findall(isnan.(data))
        datainitial[index] = 0
    end
    return datainitial
end

"""
    datafilled = emfill(data; <kwargs>)

Use the EM algorithm to fill in missing values. In each iteration:

1. fill missing values using the Kalman smoother.

2. re-estimate a `VAR` using OLS

The tolerance for convergence of the datafilled is `1e-7`, with the maximum number of iterations being 1000.

### Arguments

- `data::VecOrMat{<:Real}`: array with sample to be fitted (observations in rows, variables in columns).

### Keyword Arguments

- `P::Int64=1`: number of lags in the estimated VAR.

- `intercept::Bool=true`: whether the VAR model to be fitted contains a constant term.

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `data`  correspond to the same period.
"""
function emfill(data::VecOrMat{<:Real};
    P::Int64=1,
    intercept::Bool=true,
    fp::VecOrMat{R}=zeros(size(data))) where {R<:Real}

    (!any(isnan.(data))) && return data

    itermax = 1000
    tolerance = 1e-7
    #=
        in each iteration: 
            1. conditional expectation using kalman smoother
            2. estimate VAR from OLS
        iterate until missing values do not change too much
    =#
    T, N = size(data)
    v = setvar(0) # initiate v - not relevant
    statedata = Matrix{Float64}(undef, T, N)
    expdata = zeros(T, N)
    progress = ProgressThresh(tolerance, "Minimizing:")
    for iter in 1:itermax
        expdata = (iter == 1) ? eminitialize(data) :
                  kalmansmoother(data, v, fp=fp)
        maxv = fitols(expdata, P=P, intercept=intercept, fp=fp)
        distance = norm(expdata .- statedata, 2)
        (distance < tolerance) && break
        iter += 1
        ProgressMeter.update!(progress, distance)
        statedata = expdata
        v = maxv
        (iter == itermax) && println("Maximum number of iterations reached.")
    end

    return expdata
end

# --------------------------------------------------------------------
# BAYESIAN VAR ESTIMATION
# --------------------------------------------------------------------

"""
    VARNormalInvWishart(niw, N, P, intercept)

Constructor `VARNormalInvWishart` encases a Normal-Inverse-Wishart distribution for parameters of a vector autoregression with `N` variables, `P` lags and (maybe) an `intercept`. 
"""
struct VARNormalInvWishart
    niw::NormalInvWishart
    N::Int64
    P::Int64
    intercept::Bool
end

function unstack(beta::Vector{U},
    N::Int64,
    P::Int64,
    intercept::Bool) where {U<:Real}

    R = P * N + intercept
    B = reshape(beta, R, N) |> permutedims

    μ = intercept ? B[:, 1] : zeros(N)
    getmat(p) = B[:, intercept.+(1+N*(p-1):N*p)]
    Ψ = getmat.(1:P) |> vecmat

    return μ, Ψ
end

"""
    v = mean(d::VARNormalInvWishart)
"""
function mean(d::VARNormalInvWishart)
    b, Σ = mean(d.niw)
    μ, Ψ = unstack(b, d.N, d.P, d.intercept)
    V = setvar(μ, Ψ, Σ)
    return V
end

"""
    v = mode(d::VARNormalInvWishart)
"""
function mode(d::VARNormalInvWishart)
    b, Σ = mode(d.niw)
    μ, Ψ = unstack(b, d.N, d.P, d.intercept)
    V = setvar(μ, Ψ, Σ)
    return V
end

"""
    v = rand(d::VARNormalInvWishart)

Draw a random element from the Normal-Inverse-Wishart distribution parameterized by object `niw` of the bayesian model `d`.
"""
function rand(d::VARNormalInvWishart)
    b, Σ = rand(d.niw)
    μ, Ψ = unstack(b, d.N, d.P, d.intercept)
    V = setvar(μ, Ψ, Σ)
    return V
end


"""
    MinnesotaPrior(μ, unitroot, ψ, d, λ, γ, δ)

The `MinnesotaPrior` constructor stores parameters that form a prior to vector autoregressions similar to the so-called Minnesota Prior, introduced by `Litterman (1979)`. 

It allows two refinements: the "sum-of-coefficients" prior as in `Doan, Litterman and Sims (1984)` and the "co-persistence" prior as in `Sims (1993)`.

See `Giannone, Lenza & Primiceri (2015)` for details.

### Fields 

- `μ::Vector{Float64}`: constant term.

- `unitroot::Vector{Bool}`: select which variables are unit roots in the prior; others are white noise.

- `ψ::Vector{Float64}`: mean of the Inverse-Wishart distribution (actual IW parameter is `ψ (d-n-1)`, where `n` is the number of equations).

- `d::Float64`:  degrees of freedom of the Inverse-Wishart distribution (greater values correspond to tighter priors).

- `λ::Float64`: Minnesota prior tightness parameter (greater values correspond to looser priors).

- `γ::Float64`: sum-of-coefficients prior tightness parameter (greater values correspond to looser priors).

- `δ::Float64`: co-persistence prior tightness parameter (greater values correspond to looser priors).
"""
struct MinnesotaPrior
    μ::Vector{Float64}
    unitroot::Vector{Bool}
    ψ::Vector{Float64}
    d::Float64
    λ::Float64
    γ::Float64
    δ::Float64
end

"""
    N = size(prior::MinnesotaPrior)
"""
size(prior::MinnesotaPrior) = (length(prior.μ),)

"""
    prior = setmnprior(μ, unitroot, ψ, d, λ, γ, δ)
    prior = setmnprior(μ, ψ, d, λ, γ, δ)
    prior = setmnprior(ψ, d, λ, γ, δ)
    prior = setmnprior(ψ, d, λ)
    prior = setmnprior(ψ, λ)

Create `MinnesotaPrior` object. Constant `μ` default = zero. `unitroot` default = all variables with unit roots. `γ` and `δ` default = Inf. `d` default = number of variables + 2.
"""
setmnprior(μ, unitroot::Vector{Bool}, ψ, d, λ, γ, δ) =
    MinnesotaPrior(vect(μ), unitroot, vect(ψ), float.([d, λ, γ, δ])...)
setmnprior(μ, ψ, d, λ, γ, δ) =
    setmnprior(μ, Vector(trues(length(μ))), ψ, d, λ, γ, δ)
setmnprior(ψ, d, λ, γ, δ) = setmnprior(zeros(length(ψ)), ψ, d, λ, γ, δ)
setmnprior(ψ, d, λ) = setmnprior(ψ, d, λ, Inf, Inf)
setmnprior(ψ, λ) = setmnprior(ψ, length(ψ) + 2, λ)

function setniw(prior::MinnesotaPrior, P::Int64, intercept::Bool=true)

    μ = prior.μ
    unitroot = prior.unitroot
    ψ = prior.ψ
    d = prior.d
    λ = prior.λ

    # β average
    N = length(ψ)
    firstlagloads = diagm(0 => unitroot) |> mat
    unstackB = [firstlagloads zeros(N, N * (P - 1))]
    intercept && (unstackB = [μ unstackB])
    b = unstackB |> permutedims |> vec

    # Inverse-Wishart scale matrix parameter
    Ψ = diagm(0 => ψ * (d - N - 1))

    # Ω parameter (cov(β|Σ) = Σ ⊗ Ψ)
    S = [1 / p^2 for p in 1:P]
    Φ = [1 / ψ[j] for j in 1:N]
    Ωdiagonal = (λ^2) * kron(S, Φ)
    intercept && (Ωdiagonal = [1e6; Ωdiagonal]) # account for intercept
    Ω = diagm(Ωdiagonal)

    niw = setniw(b, Ω, Ψ, d)

    return niw
end


function sumcoef(data::VecOrMat{U},
    γ::Float64,
    P::Int64,
    unitroot::Vector{Bool},
    intercept::Bool) where {U<:Real}

    N = size(data, 2)
    y0 = mean(data[1:P, :], dims=1)[:]
    y0diag = diagm(0 => y0 ./ γ)
    ysoc = copy(y0diag)
    ysoc[.!unitroot, .!unitroot] .= 0
    xsoc = repeat(y0diag, 1, P)
    intercept && (xsoc = [zeros(N) xsoc])
    return ysoc, xsoc
end

function copers(data::VecOrMat{U},
    δ::Float64,
    P::Int64,
    unitroot::Vector{Bool},
    intercept::Bool) where {U<:Real}

    y0 = mean(data[1:P, :], dims=1)[:]
    ycop = (y0 / δ) |> permutedims
    xcop = repeat(ycop, 1, P)
    intercept && (xcop = [(1 / δ) xcop])
    ycop[.!unitroot] .= 0
    return ycop, xcop
end

"""
    marginal, posterior = bvar(data, prior; <kwargs>)

Run a Bayesian VAR regression.

### Arguments

- `data::VecOrMat{<:Real}`: array with sample (observations in rows, variables in columns).

- `prior::MinnesotaPrior`: object with parameters of the Minnesota prior.

### Output

- `marginal::Float64`: marginal likelihood of the `data`.

- `posterior::VARNormalInvWishart`: posterior distribution.

### Keyword Arguments

- `P::Int64=1`: number of lags in the VAR.

- `intercept::Bool=true`: whether the VAR model to be fitted contains a constant term.

- `fp::VecOrMat{<:Real}=0`: forcing process. A row index of `fp` and `data`  correspond to the same period.
"""
function bvar(data::VecOrMat{<:Real},
    prior::MinnesotaPrior;
    P::Int64=1,
    intercept::Bool=true,
    fp::VecOrMat{U}=zeros(size(data))) where {U<:Real}

    Nprior = size(prior)[1]
    T, N = size(data)
    @assert N == Nprior

    # dummy prior indicators
    issumcoef = (prior.γ < Inf)
    iscopers = (prior.δ < Inf)
    isdummy = issumcoef | iscopers

    # set up prior distributions
    niw = setniw(prior, P, intercept)
    y, x = regdata(data, P=P, intercept=intercept) # intercept incorporated here

    # set up dummy observations
    if issumcoef
        ysoc, xsoc = sumcoef(data, prior.γ, P, prior.unitroot, intercept)
        ydum = ysoc
        xdum = xsoc
    end

    if iscopers
        ycop, xcop = copers(data, prior.δ, P, prior.unitroot, intercept)
        ydum = issumcoef ? [ycop; ydum] : ysoc
        xdum = issumcoef ? [xcop; xdum] : xsoc
    end

    yadj = isdummy ? [ydum; y] : y
    xadj = isdummy ? [xdum; x] : x
    fpadj = isdummy ? [zeros(size(ydum)); fp[P+1:T, :]] : fp[P+1:T, :]

    # posterior distributions
    marginal, postniw = bayesreg(yadj .- fpadj, xadj, niw, intercept=false) # intercept already incorporates
    posterior = VARNormalInvWishart(postniw, N, P, intercept)
    !isdummy && return marginal, posterior

    # adjust marginal likelihood to dummy prior
    margdum = bayesreg(ydum, xdum, niw, intercept=false)[1]
    marginal = marginal - margdum

    return marginal, posterior
end

"""
    BVARSample(prior, var)

Construct with a sample of prior and main parameters. Fields `prior` and `var` must have equal lenghts. `MinnesotaPrior`s and `VAR`s stored in the same row index correspond to the same draw/observation. 

### Fields

- `prior::Vector{MinnesotaPrior}`

- `var::Vector{VAR}`

"""
struct BVARSample
    prior::Vector{MinnesotaPrior}
    var::Vector{VAR}
end



"""
    sample = rand(prior, posterior, T)

Draw a sample of VAR models from the posterior distribution of parameters. Output `sample` is a `BVARSample`, with all observations of priors equal to `prior`.

### Arguments

- `prior::MinnesotaPrior`: the prior object used to generate the posterior distribution. 

- `posterior::VARNormalInvWishart`: posterior distribution of model parameters.

- `T::Int64`: sample size.

"""
function rand(prior::MinnesotaPrior, posterior::VARNormalInvWishart, T::Int64)
    priorsample = fill(prior, T)
    varsample = Vector{VAR}(undef, T)
    progress = Progress(T, dt=0.5, barlen=25)
    for t in 1:T
        varsample[t] = rand(posterior)
        next!(progress)
    end
    sample = BVARSample(priorsample, varsample)
    return sample
end


"""
    map(f::Function, sample::BVARSample)

Apply function `f` function to each observation in `sample`. Functions must have the signature `f(::MinnesotaPrior, ::VAR)`.
"""
function map(f::Function, sample::BVARSample)
    T = length(sample.prior)
    @assert length(sample.var) == T
    output = [f(sample.prior[t], sample.var[t]) for t in 1:T]
    return output
end

"""
    path, lb, ub = irf(sample::BVARSample, shock::Vector{<:Real}, <kwargs>)
    path, lb, ub = irf(sample::BVARSample, shock::Int64, <kwargs>)

Compute the response of variables in the VARs in `sample` and report the median response, period-by-period. 
"""
function irf(sample::BVARSample, shock::Vector{U};
    T::Int64=10,
    id=1:size(sample.var[1], 1),
    cover::Float64=0.0,
    plot::Bool=true,
    options::Dict{D1,D2}=Dict(:label => id')) where {U<:Real,D1,D2}

    # sample of moving averages + plot
    N = length(id)
    S = length(sample.var)
    irfbyv = zeros(T, N, S)
    for (s, v) in enumerate(sample.var)
        maar = ma(v, T)
        maar = mapslices(m -> m * shock, maar, dims=(1, 2))
        maar = round.(maar, digits=10)

        irfbyv[:, :, s] = permutedims(maar[id, 1, :])
    end

    # median path
    path::Matrix{Float64} = median(irfbyv, dims=3) |> x -> dropdims(x, dims=3)

    lowerbo3 = mapslices(x -> quantile(x, (1 - cover) / 2), irfbyv, dims=3)
    lowerbo::Matrix{Float64} = lowerbo3[:, :, 1]

    upperbo3 = mapslices(x -> quantile(x, 1 - (1 - cover) / 2), irfbyv, dims=3)
    upperbo::Matrix{Float64} = upperbo3[:, :, 1]

    plot && irfplot(path, lowerbo, upperbo, options) |> display
    return path, lowerbo, upperbo
end

irf(sample::BVARSample,
    shock::Int64;
    T::Int64=10,
    id=1:size(sample.var[1], 1),
    cover::Float64=0.0,
    plot::Bool=true,
    options::Dict{D1,D2}=Dict(:label => id')) where {U<:Real,D1,D2} =
    irf(sample, collect(I(size(sample.var[1], 3)))[:, shock], T=T, id=id,
        cover=cover, plot=plot, options=options)
