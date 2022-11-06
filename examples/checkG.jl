include("../src/utils.jl")
# include("../ACEcore.jl/src/ACEcore.jl")

max_degree = 15
N = 1000

# 2 body case
legendre(N) = legendre_basis(N, normalize=true) 
Xs = LinRange(-1, 1, N)
Data = legendre(max_degree)(Xs)
G2b = Data' * Data
@show cond(G2b)


# 3 body case
X1 = LinRange(-1, 1, N)
X2 = LinRange(-1, 1, N)
NN = get_NN(max_degree)
poly = legendre_basis(max_degree, normalize = true)
G = getG(X1, X2, poly, NN[max_degree + 1:end])
println("cond(G): ", cond(G))



ground_pure2b = zeros(N, S - max_degree)
# see_NN = []
for i = 1:S - max_degree
    # nn, mm = NN[i + max_degree]
    #@show nn, mm
    push!(see_NN, NN[i + max_degree])
    ground_pure2b[:, i] = poly(X1)[:, nn] .* poly(X2)[:, mm] + poly(X1)[:, mm] .* poly(X2)[:, nn]
end
@show cond(ground_pure2b)