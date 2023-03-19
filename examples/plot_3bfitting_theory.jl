include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

f3_V2(xx) = 1/(1+cos(atan(xx[2]/xx[1]))^2+xx[1]^2+xx[2]^2)
plot_2D(x, y, f) = f([x, y]) 
# V2(Xs, f_V2) = [f_V2(Xs[j, :][1], Xs[j, :][2]) for j = 1:size(Xs)[1]]

E_avg(Xs, Ys, f) = [sum([f([ Xs[i,j], Ys[i,j] ]) for j=1:size(Xs[i,:])[1]])/sqrt(length(Xs[i,:])) for i = 1:size(Xs)[1]]

M = 100
max_degree = 5
max_degree_exp = 5
ord = 2 #2b+3b, can access 3b only 
body_order = :ThreeBody

testSampleSize=200
test_uniform=true
distribution=Uniform

domain_lower=-1
domain_upper=1
K_2 = 2
noise=0
# noise=1e-4

# solver = :qr
solver = :ard

# f = f1_V2
# f = f2_V2
# f = f2
f = f3_V2
Testing_func(X, Y) = E_avg(X, Y, f)
poly = legendre_basis(max_degree, normalize = true)

# D = [rand(distribution(domain_lower, domain_upper), K_2) for _=1:M]
D = rand(distribution(domain_lower, domain_upper), M, K_2, 2)
X_1 = D[:,:,:1]
X_2 = D[:,:,:2]
Theta = atan.(X_2 ./ X_1)

# X = reduce(vcat, D') # data size M x K_2
# D2 = permDist(D, ord) # generate ord needed distances pair

# J = size(D2[1])
# X_plot = X
# @show size(X_plot)

# train = rand(distribution(domain_lower, domain_upper), (M, K_2, ord))
Y = Testing_func(X_1, X_2)

A_pure, degree_list = designMatNB2D(D, poly, max_degree, max_degree_exp, ord; body = body_order)
@show cond(A_pure)

# get sol
sol_pure = solveLSQ(A_pure, Y; solver=solver)

# prediction error 

K_2_test = K_2

XX_test = rand(distribution(domain_lower, domain_upper), testSampleSize, K_2_test, 2)
X_1_test = XX_test[:,:,:1]
X_2_test = XX_test[:,:,:2]
Theta_test = atan.(X_2_test ./ X_1_test)
X_1_test_plot = reduce(vcat, X_1_test)
X_2_test_plot = reduce(vcat, X_2_test)


ground_Ep = Testing_func(X_1_test, X_2_test)

A_test, degree_list_test = designMatNB2D(XX_test, poly, max_degree, max_degree_exp, ord; body = body_order)
yp = A_test * sol_pure

# ground_yp = [f(DD_test_pair[i][j]) for i=1:testSampleSize, j=1:binomial(K_2,2)]
ground_yp = [f([X_1_test_plot[i], X_2_test_plot[i]]) for i=1:size(X_1_test_plot)[1]]

A_test_energy, degree_list_test  = designMatNB2D(XX_test, poly, max_degree,max_degree_exp, ord; body = body_order)
Ep = A_test_energy * sol_pure

println("Max degree: $max_degree, K_2: $K_2")
# println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
# println("RMSE: ", sqrt(norm(yp - ground_yp)/size(XX_test)[1]))

println("relative error of E: ", norm(Ep - ground_Ep)/norm(ground_Ep))
println("RMSE of E: ", sqrt(norm(Ep - ground_Ep)/testSampleSize))

target_x = range(domain_lower, domain_upper, length=500)
target_y = range(domain_lower, domain_upper, length=500)
plot_V2(x, y) = plot_2D.(x, y, f)

plotly();
p = plot(target_x, target_y, plot_V2, st=:surface, 
        legend = :outerbottomright,
        # xlim=[-1.1, 1.1], ylim=[-1, 2],
        # size = (1000, 800),
        alpha = 0.5,
        label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_2=$K_2")
# scatter!(X_plot[1, :], X_plot[2, :], plot_V2(X_plot[1, :], X_plot[2, :]), seriestype=:scatter, c=0, ms=0.5, label = "train")
# test_plot = reduce(hcat, reduce(hcat, DD_test_pair))'
scatter!(X_1_test, X_2_test, yp, seriestype=:scatter, c=2, ms=1, label = "prediction")
p

