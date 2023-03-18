include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

exp_dir = "results/2b1dfitting/" # result saving dir
mkpath(exp_dir)

f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3
# E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
# E_avg(X, f) = sum([f.(X[:, i])/sqrt(length(size(X)[2])) for i = 1:size(X)[2]])
E_avg(X, f) = mean([f.(X[:, i]) for i = 1:size(X)[2]]) * sqrt(length(size(X)[2]))

f = f1

ord = 1 #2b+3b, can access 3b only 
body_order = :TwoBody # 3b only

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
# MM = Int.(floor.(log.(NN) .* NN))
K_1s = [1, 4, 16, 64]
# K_1s = [1, 4]
let
    Testing_func(X) = E_avg(X, f)
    plots = []
    # push!(plots, histogram(rand(Uniform(domain_lower, domain_upper), 500), bins = 20, title="Uniform Dist"))
    for K_1 in K_1s
        error = zeros(length(NN))'
        P = plot(xaxis  = (:log, "Sample Size", ),
                            yaxis  = (:log, "RMSE"), 
                            legend = :topright, 
                            size = (300, 100))
        for t = eachindex(NN)
            M = MM[t]
            max_degree = NN[t]

            poly = legendre_basis(max_degree, normalize = true)

            X = rand(distribution(domain_lower, domain_upper), (M, K_1))
            Y = Testing_func(X) .+ noise

            A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)
            sol_pure = solveLSQ(A_pure, Y; solver=solver)
            XX_test = reshape(range(domain_lower, domain_upper, length=testSampleSize), (testSampleSize, 1))
            #XX_test = rand(distribution(domain_lower, domain_upper), (testSampleSize, 1))
                   
            

            A_test = designMatNB(XX_test, poly, max_degree, ord; body = body_order)

            yp = A_test * sol_pure
            ground_yp = Testing_func(XX_test)
            RMSE = norm(yp - ground_yp)/sqrt(M)

            println("Max Basis Deg: $max_degree")
            println("Relative error: ", norm(yp - ground_yp)/norm(ground_yp))
            println("RMSE: ", RMSE)

            error[t] = RMSE

            target_x = range(domain_lower, domain_upper, length=500)
            p = plot(target_x, f.(target_x), c=1,
            #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
                        size = (1000, 800),
                        label = "target", xlabel=L"x", ylabel=L"f(x)", title=L"K_1=%$K_1"*", Basis Size = $max_degree")
            training_flatten = reduce(vcat, X)
            test_flatten = reduce(vcat, XX_test)
            plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, ma=0.008, label = "train")
            plot!(XX_test, yp, c=2, ls=:dash, lw=2, label = "prediction")
            push!(plots, p) 
        end
        plot!(P, MM, error', lw=1, m=:o, ms=3, label=L"K_1=%$K_1")
        # plot!(P, MM, 1.0 ./ sqrt.(MM), ls=:dash,)
        push!(plots, P)
    end
    l = @layout [grid(length(K_1s), length(NN)+1)]
        
    plt = plot(plots..., layout = l, size=(2500, 1000), margin=10mm)
    savefig(plt, exp_dir*"/pp_inc_deg_[" * string(NN[1]) * "," * string(NN[end]) * "]" * "_K_1=[" * string(K_1s[1]) * "," * string(K_1s[end]) * "]" * "_order=$ord" * "_solver=$solver"*string(f))
    plt
end


