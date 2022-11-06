include("../src/utils.jl")
# include("../ACEcore.jl/src/ACEcore.jl")

X1 = LinRange(-1, 1, 1000)
X2 = LinRange(-1, 1, 1000)
max_degree = 15
NN = get_NN(max_degree)
poly = legendre_basis(max_degree, normalize = true)
G = getG(X1, X2, poly, NN[max_degree + 1:end])
println("cond(G): ", cond(G))