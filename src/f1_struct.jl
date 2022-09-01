abstract type AbstractVAR end

# promote functions
vect(x::T) where {T<:Real} = fill(float(x), 1)
vect(x::Vector{T}) where {T<:Real} = float(x)
mat(x::T) where {T<:Real} = fill(float(x), 1, 1)
mat(x::Vector{T}) where {T<:Real} = reshape(float(x), length(x), 1)
mat(x::Matrix{T}) where {T<:Real} = float(x)
mat(x::Diagonal) = Matrix(float(x))
vecmat(x::T) where {T<:Real} = [mat(x)]
vecmat(x::VecOrMat{T}) where {T<:Real} = [mat(x)]
vecmat(x::Vector{T}) where {T<:Union{Vector,Matrix}} = mat.(x)
makehermitian(X::Matrix{Float64}) = (X .+ X') / 2 |> Hermitian |> Matrix

"""
    VAR(μ, Ψ, Γ, Σ)

The `VAR` object contains the parameters of the autoregression model:

        xₜ = μ + Ψ(L) xₜ₋₁ + Γ ηₜ

with shocks independent over time and `cov(η) = Σ`.

Notation: Each element of `Ψ` corresponds to a different term of the lag 
polynomial. Rows correspond to equations. Columns correspond to loadings. 

Some functions admit a forcing process `fp`, which by default enters the VAR
on the right-hand side: `xₜ = μ + Ψ(L) xₜ₋₁ + Γ ηₜ + fpₜ`. 
"""
struct VAR <: AbstractVAR
    μ::Vector{Float64}
    Ψ::Vector{Matrix{Float64}}
    Γ::Matrix{Float64}
    Σ::Matrix{Float64}
end

function defaultgamma(N, Q)
    if N == Q
        return I(N)
    elseif N > Q
        return [I(Q); zeros(N - Q, Q)]
    elseif Q > N
        return [I(N) zeros(N, Q - N)]
    end
end

"""
    v = setvar(μ, Ψ, Γ, Σ)
    v = setvar(μ, Ψ, Σ)
    v = setvar(Ψ, Σ)
    v = setvar(Ψ)

Create `VAR` object. See documentation of `VAR` for description.
"""
setvar(μ, Ψ, Γ, Σ) = VAR(vect(μ), vecmat(Ψ), mat(Γ), mat(Σ))
setvar(μ, Ψ, Σ) = VAR(vect(μ), vecmat(Ψ),
    defaultgamma(length(μ), size(Σ, 1)), mat(Σ))
setvar(Ψ, Σ) = setvar(zeros(size(Σ, 1)), vecmat(Ψ), mat(Σ))
setvar(Ψ::Vector{T}) where {T<:VecOrMat} = setvar(vecmat(Ψ), I(size(Ψ[1], 1)))
setvar(Ψ::VecOrMat{T}) where {T<:Real} = setvar(vecmat(Ψ))
setvar(Ψ::T) where {T<:Real} = setvar(vecmat(Ψ))

"""
    N, P, Q = size(v::VAR)

### Output

- `N::Int64`: number of variables

- `P::Int64`: number of lags in the VAR

- `Q::Int64`: number of shocks
"""
function size(v::VAR)
    N = length(v.μ)
    P = length(v.Ψ)
    Q = size(v.Σ, 1)
    return N, P, Q
end
size(v::VAR, i::Int64) = size(v::VAR)[i]


# Stacked VAR: Xₜ = μ + Φ Xₜ₋₁ + Γ ηₜ
struct StackedVAR <: AbstractVAR
    μ::Vector{Float64}
    Φ::Matrix{Float64}
    Γ::Matrix{Float64}
    Σ::Matrix{Float64}
    N::Int64
end

stackvar(μ, Φ, Γ, Σ, N) = StackedVAR(vect(μ), mat.([Φ, Γ, Σ])..., N)

function stack(v::VAR)
    N, P, Q = size(v)
    Φmat = mapreduce(mat, hcat, v.Ψ)
    H = N * (P - 1) # number of additional rows to be added

    μ = [v.μ; zeros(H)]
    Φ = [Φmat; I(H) zeros(H, N)] |> Matrix
    Γ = [v.Γ; zeros(H, Q)] |> Matrix
    Σ = v.Σ
    V = stackvar(μ, Φ, Γ, Σ, N)
    return V
end

function size(V::StackedVAR)
    N = V.N
    R = length(V.μ)
    Q = size(V.Σ, 1)
    return N, R, Q
end
size(v::StackedVAR, i::Int64) = size(v)[i]

function unstack(V::StackedVAR)
    N, R, Q = size(V)
    @assert (rem(R, N) == 0) "Sizes do not conform to that of a stacked VAR"
    P = Int64(R / N)
    μ = V.μ[1:N]
    Ψ = [V.Φ[1:N, (1+N*(p-1)):(N*p)] for p in 1:P]
    Γ = V.Γ[1:N, :]
    Σ = V.Σ
    v = setvar(μ, Ψ, Γ, Σ)
    return v
end

function stackfp(fp::VecOrMat{<:Real}, v::VAR)
    N, P, Q = size(v)
    H = N * (P - 1) # number of columns to be added to fp
    T = size(fp, 1)
    FP = [fp zeros(T, H)]
    return FP
end