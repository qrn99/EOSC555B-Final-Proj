include("../src/HelperFunctions.jl")
include("../src/utils.jl")

using .HelperFunctions
using LaTeXStrings, Interact

f1(x) = abs(x)^3
f2(x) = 1/(1+8*x^2)
E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]

##
# @manipulate for BasisDeg = [5, 10, 20, 40, 80],  M = [40, 80, 160, 200], 
#     testSampleSize=[20, 50, 100], adaptedTrainSize=[100, 150, 200], 
#     test_uniform=[false, true],
#     rdf_bimodal = range(0.15, 0.3, 16), K_R = [1, 4, 16, 64], noise=[0.0, 1e-4, 1e-3] #, 1e-2, 1e-1]

BasisDeg=20
M=40
testSampleSize=400
test_uniform=true
adaptedTrainSize=testSampleSize

domain_lower=-1
domain_upper=1
K_R = 4
noise=1e-4

let fs=[f1, f2]
    N = BasisDeg

    # Add poly when pack has it
    basis_choices = [HelperFunctions.cheb, HelperFunctions.legendre, "OrthPoly"]

    distributions = ["uniform", "hole", "bimodal", "abs"]

    plots = []

    data_dst = HelperFunctions.generate_data_dst(0.3, -0.8, 0.9, domain_lower, domain_upper; abs_width=0.8)

    push!(plots, histogram(rand(1000)*2 .- 1, bins = 20, title="Uniform Dist"))

    Random.seed!(12216859)
    for each in keys(data_dst)
        push!(plots, histogram(rand(data_dst[each], 1000), bins = 20, title="$each Dist"))
    end

    for f in fs
        for basis_choice in basis_choices
            for dst in distributions 
                train, test = HelperFunctions.generate_data(M, K_R, testSampleSize, data_dst, dst)

                training_flatten = sort(collect(Iterators.flatten(train)))
    #                     xp_pp_test = sort(collect(Iterators.flatten(test)))
                xp_pp_test = range(domain_lower, domain_upper, length=testSampleSize)

                basis = HelperFunctions.get_basis(basis_choice, BasisDeg, K_R, data_dst, dst; adaptedTrainSize=adaptedTrainSize)
                
                μ = HelperFunctions.lr(E, f, train, basis, N; noise=noise)
                yp = HelperFunctions.predict(xp_pp_test, basis, μ)

                target_x = range(-1, 1, length=200)
                p = plot(target_x, f.(target_x), c=1,
    #                         xlim=[-1.1, 1.1], ylim=[-1, 2],
                    label = "target", xlabel="x", ylabel="pair potential", title="$basis_choice with dst=$dst")

                plot!(training_flatten, f.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, label = "")
                plot!(xp_pp_test, yp, c=2, ls=:dash, label = "prediction")
                push!(plots, p)
            end
        end
    end
    l = @layout [grid(length(fs)+length(basis_choices)+2, length(distributions))]
    plot(plots..., layout = l, size=(2500, 1500), plot_title="deg=$BasisDeg, K_R = $K_R, noise=$noise, test_uniform=$test_uniform")
end
##

let
    M = 100
    BasisDeg = 10
    N = BasisDeg
    testSampleSize = 500
    K_R=4

    # f2(x) = x^2 - 10 * cos(2*pi*x) + 10
    f1(x) = 1/(1+8*x^2)
    f2(x) = abs(x)^3
    E_avg(X, f) = sum([f.(X[:, i]) for i = 1:size(X)[2]])
     
    Testing_func(X) = E_avg(X, f1) + E_avg(X, f2)
    solver = :qr
    max_degree = BasisDeg
    N = max_degree
    # ord = 1 # 2body ord = 1 = body - 1
    ord = 2 # 3body ord = 2 = body - 1
    NN = get_NN(max_degree, ord)
    NN2b = NN[length.(NN) .== 1] # for analysis
    NN3b = NN[length.(NN) .== 2]
    poly = legendre_basis(max_degree, normalize = true)

    X = rand(distribution(domain_lower, domain_upper), (M, K_R))

    # initialize design matrix
    A_pure = zeros(M, length(NN))
    B = Testing_func(X)

    # for evaluating ground truth
    poly_list = [poly(X[:, i]) for i = 1:K_R]

    for i = 1:length(NN2b)
        nn = NN2b[i]
        A_pure[:, i] = sum([PX1[:, nn] for PX1 in poly_list])
    end

    for i = 1:length(NN3b)
        nn, mm = NN3b[i]
        A_pure[:, length(NN2b) + i] = sum([PX1[:, nn] .* PX2[:, mm] for PX1 in poly_list for PX2 in poly_list if PX1 != PX2])
    end
    

    if solver == :qr
        # solve the problem with qr
        LL = length(NN)
        λ = 0.1
        sol_pure = qr(vcat(A_pure, λ * I(LL) + zeros(LL, LL))) \ vcat(B, zeros(LL))
    elseif solver == :ard
    # solve the problem with ARD       
        ARD = pyimport("sklearn.linear_model")["ARDRegression"]
        clf = ARD()
        sol_pure = clf.fit(A_pure, B).coef_
    end
       
    
    XX_test = range(-1, 1, length=testSampleSize)
    
    A_test = zeros(testSampleSize, length(NN))

    basis = poly(XX_test)
    A_test[:, 1:length(NN2b)] = basis

    for i = 1:length(NN3b)
        nn, mm = NN3b[i]
        A_test[:, length(NN2b) + i] = sum([basis[:, nn] .* basis[:, mm]])
    end

    yp = A_test * sol_pure
    ground_yp = f1.(XX_test) + f2.(XX_test)

    println("relative error of pure basis: ", norm(yp - ground_yp)/norm(ground_yp))
    println("RMSE: ", norm(yp - ground_yp)/sqrt(M))
    
#    target_x = rand(distribution(domain_lower, domain_upper), (300, K_R))
    target_x = range(domain_lower, domain_upper, length=400)
    p = plot(target_x, f1.(target_x)+f2.(target_x), c=1,
#                         xlim=[-1.1, 1.1], ylim=[-1, 2],
                label = "target", xlabel="x", ylabel="f(x)", title="order=$ord, basis maxdeg = $max_degree, sample size = $M, K_R=$K_R")
    training_flatten = reduce(vcat, X)
    test_flatten = reduce(vcat, XX_test)
    plot!(training_flatten, f1.(training_flatten)+f2.(training_flatten), c=1, seriestype=:scatter, m=:o, ms=1, label = "")
    plot!(XX_test, yp, c=2, ls=:dash, label = "prediction")
    p
end
##