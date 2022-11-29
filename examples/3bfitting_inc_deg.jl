include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3
# E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
E_avg(X, f) = sum([f.(X[:, i])/length(size(X)[2]) for i = 1:size(X)[2]])

f = f1

ord = 2 #2b+3b, can access 3b only 
body_order = :ThreeBody # 3b only

testSampleSize=400
test_uniform=true
adaptedTrainSize=testSampleSize
distribution=Uniform

domain_lower=-1
domain_upper=1

noise=0
# noise=1e-4

solver = :qr
# solver = :ard

NN = [5, 10, 20, 30]
MM = 10*NN.^2 .+ 50
K_Rs = [2, 4, 12]
let
    Testing_func(X) = E_avg(X, f)
    plots = []
    # push!(plots, histogram(rand(Uniform(domain_lower, domain_upper), 500), bins = 20, title="Uniform Dist"))
    for K_R in K_Rs
        error = zeros(length(NN))'
        P = plot(xaxis  = (:log, "sample size", ),
                            yaxis  = (:log, "RMSE"), 
                            legend = :outerbottomright, 
                            size = (300, 100))
        for t = eachindex(NN)
            M = MM[t]
            max_degree = NN[t]

            poly = legendre_basis(max_degree, normalize = true)

            X = rand(distribution(domain_lower, domain_upper), (M, K_R))
            Y = Testing_func(X) .+ noise

            A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)

            sol_pure = solveLSQ(A_pure, Y; solver=solver)
                    
            XX_test = range(domain_lower, domain_upper, length=testSampleSize)

            A_test = predMatNB(XX_test, poly, max_degree, ord; body = body_order)
            yp = A_test * sol_pure
            ground_yp = f.(XX_test)
            RMSE = norm(yp - ground_yp)/sqrt(M)

            println("Max Basis Deg: $max_degree")
            println("Relative error: ", norm(yp - ground_yp)/norm(ground_yp))
            println("RMSE: ", RMSE)

            error[t] = RMSE

            target_x = range(domain_lower, domain_upper, length=500)
            p = plot(target_x, f.(target_x), c=1,
            #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
                        size = (1000, 800),
                        label = "target", xlabel="x", ylabel="f(x)", title="maxdeg = $max_degree, sample size = $M")
            training_flatten = reduce(vcat, X)
            test_flatten = reduce(vcat, XX_test)
            plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, label = "")
            plot!(XX_test, yp, c=2, ls=:dash, lw=2, label = "prediction")
            push!(plots, p) 
        end
        plot!(P, MM, error', lw=1, m=:o, ms=3, label="K_R=$K_R")
        push!(plots, P)
    end
    l = @layout [grid(length(K_Rs), length(NN)+1)]
        
    plot(plots..., layout = l, size=(2500, 1000), margin=10mm, plot_title="order=$ord, solver=$solver, noise=$noise, test_uniform=$test_uniform")
end


