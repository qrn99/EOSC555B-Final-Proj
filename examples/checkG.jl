include("../src/utils.jl")
# include("../ACEcore.jl/src/ACEcore.jl")
using Distributions
using Plots

max_degree = 15
N = 10000

# 2 body case
legendre(N) = legendre_basis(N, normalize=true) 
Xs = LinRange(-1, 1, N)
Data = legendre(max_degree)(Xs)
G2b = Data' * Data
@show cond(G2b)


# 3 body case
NN = get_NN(max_degree)
poly = legendre_basis(max_degree, normalize = true)
X1, X2 = rand(Uniform(-1, 1), N), rand(Uniform(-1, 1), N)
S = length(NN)
ground_pure2b = zeros(N, S - max_degree)
poly_X1 = poly(X1)
poly_X2 = poly(X2)
# see_NN = []
for i = 1:S - max_degree
    nn, mm = NN[i + max_degree]
    ground_pure2b[:, i] = (poly_X1[:, nn] .* poly_X2[:, mm] + poly_X1[:, mm] .* poly_X2[:, nn])
end
M = ground_pure2b
@show cond(ground_pure2b)

# plot the matrix 
heatmap(ground_pure2b' * ground_pure2b , aspect_ratio=1)
savefig("check_ground_pure2b")