include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact

r_in = 0.85 # with envelope should be 0.0
r_nn = 1.0
r_cut = 3.0

# hyperparameter
α = 3   # morse coordinate change
p = 4   # envelope
q = 4   # prior smoothness, could be exp

# # pair potential
# ϕ(r) = r^(-12) - 2*r^(-6)

# morse transform
# x1(r) = 1 / (1 + r / r_nn)

#Agnesis Transform
# 0.33 = a = (p-1)/(p+1)
x1(r) = 1.0 / (1.0 + 0.33*(r / r_nn)^2)

x_in = x1(0.0); x_cut = x1(r_cut) # Regularize til r=0
x(r) = 2 * (x1(r) - x_cut) / (x_in - x_cut) - 1

# envelope
env(r) = (r^(-p) - r_cut^(-p) + p * r_cut^(-p-1) * (r - r_cut)) * (r < r_cut)

let
    f1(x) = 1/(1+8*x^2)
    f2(x) = abs(x)^3
    ϕ(r) = r^(-12) - 2*r^(-6)

    # E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
    E_avg(X, f) = sum([f.(X[:, i])/length(size(X)[2]) for i = 1:size(X)[2]])

    M = 200
    max_degree = 20
    ord = 1 #2b+3b, can access 3b only 
    body_order = :TwoBody

    testSampleSize=500
    test_uniform=true
    distribution=Uniform

    domain_lower=r_in
    domain_upper=r_cut
    K_R = 4

    noise=0
    # noise=1e-4

    solver = :qr

    f = ϕ
    Testing_func(X) = E_avg(X, f)
    poly = legendre_basis(max_degree, normalize = true)

    # X = rand(distribution(domain_lower, domain_upper), (M, K_R))

    #--- random pos of atom ---#
    # dim = 3, K_R = num_of_atoms
    X = reduce(hcat, [pos_to_dist(gen_correlated_pos(Uniform(-10, 10), 3, K_R), 2) for _=1:M])'
    #--- end ---#

    #--- atomic strucutres ---#
    # # concat pos into an matrix where each row is one atom pos
    # train_pos = []
    # for i=1:M
    #     at, r_nn, R, Eref = atom_bulk()
    #     push!(train_pos, R) 
    # end
    # X = [pos_to_dist(pos, 2) for pos in train_pos]
    # X = reduce(hcat, X)
    # #--- end ---#

    Y = Testing_func(X) .+ noise

    A_pure = designMatNB(x.(X), poly, max_degree, ord; body = body_order)

    sol_pure = solveLSQ(A_pure, Y; solver=solver)

    # XX_test = range(domain_lower, domain_upper, length=testSampleSize)
    # A_test = predMatNB(XX_test, poly, max_degree, ord; body = body_order)

    XX_test = rand(distribution(domain_lower, domain_upper), (testSampleSize, 1))
    XX_test = XX_test[sortperm(XX_test[:, 1]), :]

    A_test = designMatNB(x.(XX_test), poly, max_degree, ord; body = body_order)

    yp = A_test * sol_pure
    ground_yp = f.(XX_test)

    println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
    println("RMSE: ", norm(yp - ground_yp)/sqrt(M))

    target_x = range(domain_lower, domain_upper, length=500)
    p = plot(target_x, f.(target_x), c=1,
    #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
                # size = (1000, 800),
                label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
    training_flatten = reduce(vcat, X)
    test_flatten = reduce(vcat, XX_test)
    plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, label = "")
    plot!(XX_test, yp, c=2, ls=:dash, label = "prediction")
    p
end
##