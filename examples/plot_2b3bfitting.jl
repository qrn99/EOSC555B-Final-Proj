include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact, PyCall
##
f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3
# E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
E_avg(X, f) = sum([f.(X[:, i])/length(size(X)[2]) for i = 1:size(X)[2]])

f1_V2(xx) = xx[1]^2 + xx[2]^2
f2_V2(xx) = xx[1]^2 - 10 * cos(2*pi*xx[2]^2)
plot_2D(x, y, f) = f([x, y]) 
E_avg_2D(Xs, f) = [sum([f(xx) for xx in Xs[i]])/length(Xs[i]) for i = 1:size(Xs)[1]]

##
M = 500
testSampleSize = 400
max_degree = 15
ord = 2 #2b+3b
body_order = :TwoBodyThreeBody

testSampleSize=400
test_uniform=true
distribution=Uniform

domain_lower=-1
domain_upper=1
K_R = 2

noise=0
# noise=1e-4

# solver = :qr
solver = :ard

# f_2b = f1
f_2b = f2
f_3b = f1_V2
# f_3b = f2_V2

Testing_func(X, D) = E_avg(X, f_2b) + E_avg_2D(D, f_3b)
poly = legendre_basis(max_degree, normalize = true)

X = rand(distribution(domain_lower, domain_upper), (M, K_R))
D = [X[i, :] for i in 1:M]
D2 = permDist(D, ord)
X_plot = reduce(hcat, reduce(hcat, D2))
Y = Testing_func(X, D2) .+ noise

A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)

sol_pure = solveLSQ(A_pure, Y; solver=solver) 


#------ Test 2b ------#
XX_test_2b = rand(distribution(domain_lower, domain_upper), (testSampleSize, 1))
XX_test_2b = XX_test_2b[sortperm(XX_test_2b[:, 1]), :]

A_test_2b = designMatNB(XX_test_2b, poly, max_degree, 1; body = :TwoBody)
yp_2b = A_test_2b * sol_pure[1:max_degree]
ground_yp_2b = f_2b.(XX_test_2b)


println("relative error for 2b: ", norm(yp_2b - ground_yp_2b)/norm(ground_yp_2b))
println("RMSE for 2b: ", norm(yp_2b - ground_yp_2b)/sqrt(testSampleSize))

target_x = range(domain_lower, domain_upper, length=500)
p1 = plot(target_x, f_2b.(target_x), c=1,
#                         xlim=[-1.1, 1.1], ylim=[-1, 2],
            # size = (1000, 800),
            label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
training_flatten = reduce(vcat, X)
test_flatten = reduce(vcat, XX_test_2b)
plot!(training_flatten, f_2b.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, label = "")
plot!(XX_test_2b, yp_2b, c=2, ls=:dash, label = "prediction")

#------ Test 3b ------#
# prediction error 
DD_test_3b = [rand(distribution(domain_lower, domain_upper), K_R) for _=1:testSampleSize]
XX_test_3b = rand(distribution(domain_lower, domain_upper), (testSampleSize, 2))

# XX_test = reduce(vcat, DD_test')
# DD_test2D = permDist(DD_test, ord) # generate ord needed distances pair
# DD_test2D = DD_test2D[sortperm(DD_test2D[:, 1]), :]
# DD_test2D = reduce(vcat, DD_test2D')
# DD_test2D_plot = reduce(vcat, DD_test2D)

A_test = designMatNB(XX_test_3b, poly, max_degree, 2; body = :ThreeBody)

yp_3b = A_test * sol_pure[max_degree+1:end]
ground_yp_3b = [f_3b(XX_test_3b[i, :]) for i=1:testSampleSize]

println("relative error for 3b: ", norm(yp_3b - ground_yp_3b)/norm(ground_yp_3b))
println("RMSE for 3b: ", sqrt(norm(yp_3b - ground_yp_3b)/testSampleSize))

target_x = range(domain_lower, domain_upper, length=500)
target_y = range(domain_lower, domain_upper, length=500)
plot_V2(x, y) = plot_2D.(x, y, f_3b)

plotly();
p2 = plot(target_x, target_y, plot_V2, st=:surface, 
        legend = :outerbottomright,
        # xlim=[-1.1, 1.1], ylim=[-1, 2],
        # size = (1000, 800),
        alpha = 0.5,
        label = "target", xlabel="x", ylabel="f(x)", title="")
scatter!(X_plot[1, :], X_plot[2, :], plot_V2(X_plot[1, :], X_plot[2, :]), seriestype=:scatter, c=0, ms=0.5, label = "train")
scatter!(XX_test_3b[:, 1], XX_test_3b[:, 2], yp_3b, seriestype=:scatter, c=2, ms=1, label = "prediction")

p = plot(p1, p2, margin=10mm, size=(2000, 1500), legend=:outerbottom, plot_title="order=$body_order, solver=$solver, noise=$noise")
##