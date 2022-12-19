include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact
using PyCall

exp_dir = "results/3b1dfitting/" # result saving dir
mkpath(exp_dir)

f1_V2(xx) = xx[1]^2 + xx[2]^2
f2_V2(xx) = xx[1]^2 - 10 * cos(2*pi*xx[2]^2)
plot_2D(x, y, f) = f([x, y]) 
# V2(Xs, f_V2) = [f_V2(Xs[j, :][1], Xs[j, :][2]) for j = 1:size(Xs)[1]]

E_avg(Xs, f) = [sum([f(xx) for xx in Xs[i]])/length(Xs[i]) for i = 1:size(Xs)[1]]

f = f1_V2

ord = 2 #2b+3b, can access 3b only 
body_order = :ThreeBody

testSampleSize=400
test_uniform=true
adaptedTrainSize=testSampleSize
distribution=Uniform

domain_lower=-1
domain_upper=1

noise=0
# noise=1e-4

solver = :qr
solver = :ard

NN = [5, 10, 20]
MM = NN.^2
K_Rs = [3, 6, 12]
let
    f = f1_V2
    # f = f2_V2
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
            D = [rand(distribution(domain_lower, domain_upper), K_R) for _=1:M]
            X = reduce(vcat, D') # data size M x K_R
            D2 = permDist(D, ord) # generate ord needed distances pair
            J = size(D2[1])
            X_plot = reduce(hcat, reduce(hcat, D2))

            Y = Testing_func(D2)
            
            A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)
            
            @show cond(A_pure)

            # get sol
            sol_pure = solveLSQ(A_pure, Y; solver=solver)

            # prediction error 
            # predict two dist clusters
            XX_test = rand(distribution(domain_lower, domain_upper), (testSampleSize, 2))
            DD_test_pair = [[XX_test[i, :]] for i=1:testSampleSize]

            Ep = Testing_func(DD_test_pair)

            A_test = designMatNB(XX_test, poly, max_degree, ord; body = body_order)
            yp = A_test * sol_pure

            ground_yp = [f(XX_test[i, :]) for i=1:testSampleSize]

            Norm_diff = norm(yp - ground_yp)
            RMSE = Norm_diff/sqrt(testSampleSize)
            println("Max Basis Deg: $max_degree, K_R: $K_R")
            println("relative error of pure basis: ", Norm_diff/norm(ground_yp))
            println("RMSE: ", RMSE)
            # println("Training error: ", norm((A_pure * sol_pure - Y))/ norm(Y))
            # println("max sol pure: ", maximum(abs.(sol_pure)))
            # println("max A pure: ", maximum(abs.(A_pure)))
            # println("max pred: ", maximum(abs.(yp)))
            # println("max ground: ", maximum(abs.(Ep)))
            # println("mean ground: ", mean(Ep))
            println("relative error of E: ", norm(yp - Ep)/norm(Ep))
            println("RMSE of E: ", norm(yp - Ep)/sqrt(testSampleSize))

            target_x = range(domain_lower, domain_upper, length=500)
            target_y = range(domain_lower, domain_upper, length=500)
            plot_V2(x, y) = plot_2D.(x, y, f)

            error[t] = RMSE

            target_x = range(domain_lower, domain_upper, length=500)
            plotly();
            p = plot(target_x, target_y, plot_V2, st=:surface, 
                    legend = :outerbottomright,
                    # xlim=[-1.1, 1.1], ylim=[-1, 2],
                    # size = (1000, 800),
                    alpha = 0.5,
                    label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
            scatter!(X_plot[1, :], X_plot[2, :], plot_V2(X_plot[1, :], X_plot[2, :]), seriestype=:scatter, c=0, ms=0.5, label = "train")
            test_plot = reduce(hcat, reduce(hcat, DD_test_pair))'
            scatter!(test_plot[:, 1], test_plot[:, 2], yp, seriestype=:scatter, c=2, ms=1, label = "prediction")
            push!(plots, p)
        end
        plot!(P, MM, error', lw=1, m=:o, ms=3, label="K_R=$K_R")
        push!(plots, P)
    end
    l = @layout [grid(length(K_Rs), length(NN)+1)]
        
    plt = plot(plots..., layout = l, 
                size=(2500, 1000), 
                margin=10mm, plot_title="order=$ord, solver=$solver, noise=$noise, test_uniform=$test_uniform")
    savefig(plt, exp_dir*"/3b1d_inc_deg_[" * string(NN[1]) * "," * string(NN[end]) * "]" * "_K_R=[" * string(K_Rs[1]) * "," * string(K_Rs[end]) * "]")
    plt
end


