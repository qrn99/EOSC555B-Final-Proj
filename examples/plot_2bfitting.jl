include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact

##
f1(x) = abs(x)^3
f2(x) = 1/(1+8*x^2)
E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/sqrt(length(R)) for R in Rs ]
exp_dir = "results/2bfitting/" # result saving dir
mkpath(exp_dir)
# ##
# # @manipulate for BasisDeg = [5, 10, 20, 40, 80],  M = [40, 80, 160, 200], 
# #     testSampleSize=[20, 50, 100], adaptedTrainSize=[100, 150, 200], 
# #     test_uniform=[false, true],
# #     rdf_bimodal = range(0.15, 0.3, 16), K_N = [1, 4, 16, 64], noise=[0.0, 1e-4, 1e-3] #, 1e-2, 1e-1]

BasisDeg=20

num_data = 1000
K_N = 5

M = floor(num_data / K_N)

testSampleSize=400
test_uniform=true
adaptedTrainSize=testSampleSize

domain_lower=-1
domain_upper=1
noise=1e-4

let fs=[f1, f2]
    N = BasisDeg

    # Add poly when pack has it
    basis_choices = [HelperFunctions.cheb, HelperFunctions.legendre, "orthpoly"]

    distributions = ["uniform", "hole", "bimodal", "abs"]

    plots = []

    data_dst = HelperFunctions.generate_data_dst(0.3, -0.8, 0.9, domain_lower, domain_upper; abs_width=0.8)

    push!(plots, histogram(rand(1000)*2 .- 1, bins = 20, title="Uniform Dist", legend = false))

    Random.seed!(12216859)
    for each in keys(data_dst)
        push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, title="$each Dist", legend = false))
    end

    for f in fs
        for basis_choice in basis_choices
            for dst in distributions 
                train, test = HelperFunctions.generate_data(M, K_N, testSampleSize, data_dst, dst)
                # @show size(train)
                training_flatten = sort(collect(Iterators.flatten(train)))
    #                     xp_pp_test = sort(collect(Iterators.flatten(test)))
                #@show size(training_flatten)
                xp_pp_test = range(domain_lower, domain_upper, length=testSampleSize)
                basis = HelperFunctions.get_basis(basis_choice, BasisDeg, K_N, data_dst, dst; adaptedTrainSize=adaptedTrainSize)
                
                μ = HelperFunctions.lr(E, f, train, basis, N; noise=noise)
                yp = HelperFunctions.predict(xp_pp_test, basis, μ)
                target_x = range(-1, 1, length=200)
                p = plot(target_x, f.(target_x), c=1, lw=2,
    #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
                    label = "target", xlabel="x", ylabel="f(x)", title="$basis_choice with $dst distribtuion")

                plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, ma=0.08, label = "train")
                plot!(xp_pp_test, yp, c=2, lw=2, ls=:dash, label = "prediction")
                push!(plots, p)
            end
        end
    end
    l = @layout [grid(length(fs)+length(basis_choices)+2, length(distributions))]
    PL = plot(plots..., layout = l, margin = 5mm, size=(2600, 1500), plot_title="J=$BasisDeg, K_N = $K_N, noise=$noise, test_uniform=$test_uniform")
    # savefig(PL, exp_dir*"2b_basic_with_train_plots_KN=" * string(K_N)*"f=$fs")
    # savefig(PL, exp_dir*"2b_basic_plots_KN=" * string(K_N)*"f=$fs")
end

# condition number 
let 
    N = BasisDeg
    let K_Ns = [1, 4, 16, 64], Ms = 2 .^(7:13)
        testSampleSize = 10 # does not matter for design matrix
        # Add poly when pack has it
        basis_choices = [cheb, legendre, "orthpoly"]
    
        distributions = ["uniform"]
    
        plots = []
    
        data_dst = HelperFunctions.generate_data_dst(0.3, -0.8, 0.9, domain_lower, domain_upper; abs_width=0.8)

#         push!(plots, histogram(rand(1000)*2 .- 1, bins = 20, title="Uniform Dist"))
    
#         Random.seed!(12216859)
#         for each in keys(data_dst)
#             push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, title="$each Dist"))
#         end
        
        for K_N in K_Ns
            for dst in distributions 
                P = plot(xaxis  = (:log, "Sample Size (M)"),
                            yaxis  = (:log, L"\kappa(A^TA)"), 
                            legend = :outerbottomright, 
                            title = "K_N=$K_N",
                            size = (200, 200))
                plot!(P, Ms, 1 .^(7:13), c=1, ls=:dash, label="")

                for basis_choice in basis_choices
#                     basis = get_basis(basis_choice, N, K_N, data_dst, dst; adaptedTrainSize=adaptedTrainSize)
                    conds = zeros(1, length(Ms))
                    for i in 1:length(Ms)
                        M = Ms[i]
                        # give sufficient data for adapted basis
                        basis = HelperFunctions.get_basis(basis_choice, N, K_N, data_dst, dst; adaptedTrainSize=M)
                        rs, _ = HelperFunctions.generate_data(M, K_N, testSampleSize, data_dst, dst)
                        A = HelperFunctions.design_matrix(rs, basis, N)
                        conds[i] = cond(A'A)
                    end
                    plot!(P, Ms, conds', xscale=:log10, yscale=:log10, shape=:circle, label="$basis_choice")
                end
                push!(plots, P)
            end
        end
        l = @layout [grid(1,4)]
        # plot_title="Condition Number of the Gram Matrix with Basis Size=$N"
        p1 = plot(plots..., layout = l, size=(2000, 400), margin = 10mm)
        # savefig(p1, exp_dir*"basic_model_cond_no_deg=$N"*"dst=$dst")
        # p1
    end

end
##

# let
#     f1(x) = 1/(1+8*x^2)
#     f2(x) = abs(x)^3
#     # E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
#     E_avg(X, f) = sum([f.(X[:, i])/length(size(X)[2]) for i = 1:size(X)[2]])

#     M = 100
#     max_degree = 20
#     ord = 1 #2b+3b, can access 3b only 
#     body_order = :TwoBody

#     testSampleSize=400
#     test_uniform=true
#     distribution=Uniform

#     domain_lower=-1
#     domain_upper=1
#     K_N = 4

#     noise=0
#     # noise=1e-4

#     solver = :qr

#     f = f2
#     Testing_func(X) = E_avg(X, f)
#     poly = legendre_basis(max_degree, normalize = true)

#     X = rand(distribution(domain_lower, domain_upper), (M, K_N))
#     Y = Testing_func(X) .+ noise

#     A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)

#     sol_pure = solveLSQ(A_pure, Y; solver=solver)

#     XX_test = range(domain_lower, domain_upper, length=testSampleSize)

#     A_test = predMatNB(XX_test, poly, max_degree, ord; body = body_order)
#     yp = A_test * sol_pure
#     ground_yp = f.(XX_test)

#     println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
#     println("RMSE: ", norm(yp - ground_yp)/sqrt(M))

#     target_x = range(domain_lower, domain_upper, length=500)
#     p = plot(target_x, f.(target_x), c=1,
#     #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
#                 size = (1000, 800),
#                 label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_N=$K_N")
#     training_flatten = reduce(vcat, X)
#     test_flatten = reduce(vcat, XX_test)
#     plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, label = "")
#     plot!(XX_test, yp, c=2, ls=:dash, label = "prediction")
#     p
# end
# ##
