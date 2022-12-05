include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3

f1_V2(xx) = xx[1]^2 + xx[2]^2
f2_V2(xx) = xx[1]^2 - 10 * cos(2*pi*xx[2]^2)
plot_2D(x, y, f) = f([x, y]) 
# V2(Xs, f_V2) = [f_V2(Xs[j, :][1], Xs[j, :][2]) for j = 1:size(Xs)[1]]

E_avg(Xs, f) = [sum([f(xx) for xx in Xs[i]])/length(Xs[i]) for i = 1:size(Xs)[1]]

M = 200
max_degree = 15
ord = 2 #2b+3b, can access 3b only 
body_order = :ThreeBody

testSampleSize=400
test_uniform=true
distribution=Uniform

domain_lower=-1
domain_upper=1
K_R = 4

noise=0
# noise=1e-4

solver = :qr

f = f1_V2
Testing_func(X) = E_avg(X, f)
poly = legendre_basis(max_degree, normalize = true)

D = [rand(distribution(domain_lower, domain_upper), K_R) for _=1:M]
X = reduce(vcat, D')
D2 = permDist(D, ord) # generate ord needed distances pair

J = size(D2[1])
X_plot = reduce(hcat, reduce(hcat, D2))

# train = rand(distribution(domain_lower, domain_upper), (M, K_R, ord))
Y = Testing_func(D2)

A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)
@show cond(A_pure)

# get sol
sol_pure = solveLSQ(A_pure, Y; solver=solver)

# prediction error 
DD_test = [rand(distribution(domain_lower, domain_upper), K_R) for _=1:testSampleSize]
XX_test = rand(distribution(domain_lower, domain_upper), (testSampleSize, 2))

# XX_test = reduce(vcat, DD_test')
# DD_test2D = permDist(DD_test, ord) # generate ord needed distances pair
# DD_test2D = DD_test2D[sortperm(DD_test2D[:, 1]), :]
# DD_test2D = reduce(vcat, DD_test2D')
# DD_test2D_plot = reduce(vcat, DD_test2D)

A_test = designMatNB(XX_test, poly, max_degree, ord; body = body_order)
yp = A_test * sol_pure

ground_yp = [f(XX_test[i, :]) for i=1:testSampleSize]

println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
println("RMSE: ", sqrt(norm(yp - ground_yp)/testSampleSize))

target_x = range(domain_lower, domain_upper, length=500)
target_y = range(domain_lower, domain_upper, length=500)
plot_V2(x, y) = plot_2D.(x, y, f)

plotly();
p = plot(target_x, target_y, plot_V2, st=:surface, 
        legend = :outerbottomright,
        # xlim=[-1.1, 1.1], ylim=[-1, 2],
        # size = (1000, 800),
        alpha = 0.5,
        label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
scatter!(X_plot[1, :], X_plot[2, :], plot_V2(X_plot[1, :], X_plot[2, :]), seriestype=:scatter, c=0, ms=0.5, label = "train")
scatter!(XX_test[:, 1], XX_test[:, 2], yp, seriestype=:scatter, c=2, ms=1, label = "prediction")
p

