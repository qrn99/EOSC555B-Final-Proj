include("../src/utils.jl")
using PyCall

# -------------------------------------------------------testing functionss------------------------------------------#
function Rastrigin(X)
    return sum([X[:, i].^2 - 10 * cos.(2 * pi * X[:, i]) .+ 10 for i = 1:size(X)[2]])
end

f1(x) = 1/(1+8*x^2)
f2(x) = abs(x)^3
f3(x) = x^2 - 10*cos(2*pi*x)+10

E_avg(X, f) = sum([f.(X[:, i])/length(size(X)[2]) for i = 1:size(X)[2]])

# -------------------------------------------------------testing configs------------------------------------------#
Random.seed!(1255534)
max_degree = 15 # max_degree of polynomial
exp_dir = "results/test_result_3b_Rastrigin_dnumsam_samedegree/" # result saving dir
mkpath(exp_dir)
distribution = Uniform # determinated by type of basis used
# solver = :qr # solver, currently support qr and ARD
solver = :ard
num_sam_list = 10 .^(2:5) # number of data used in training
num_test = 1000 # number of data used in test, will be drawn from the same distribution as training
aa = num_sam_list[1]
bb = num_sam_list[end]
testSampleSize = 500
ord = 2 # ord = body - 1
body_order = :TwoBodyThreeBody
K_R = 5 # number of radius distances in a configuration
poly = legendre_basis(max_degree, normalize = true) # polynomial basis used

f = f3
Testing_func(X) = E_avg(X, f) # testing function for evaluating basis
plotting = true # decide to save result or not

# -------------------------------------------------------basic constructions------------------------------------------#
NN = get_NN(max_degree, ord)
NN2b = NN[length.(NN) .== 1] # for analysis
NN3b = NN[length.(NN) .== 2]

pure_err = []
err_f = []
cond_num_2b_pure = []
cond_num_3b_pure = []

# -------------------------------------------------------testing loop------------------------------------------#
for num_sam in num_sam_list
    pure_err_curr = 0
    err_f_curr = 0
    cond_num_2b_pure_curr = 0
    cond_num_3b_pure_curr = 0
    
    for i = 1:10
        num_sam = Integer(num_sam)
        println("----------------------------------------------")
        println("current number of sample points: ", num_sam)
        path = exp_dir * "maxdeg="*string(max_degree)*"/num_sam="*string(num_sam)*"/"
        mkpath(path)
        X = rand(distribution(-1, 1), (num_sam, K_R))

        # initialize design matrix
        Y = Testing_func(X)

        # for evaluating ground truth
        poly_list = [poly(X[:, i]) for i = 1:K_R]

        A_pure = designMatNB(X, poly, max_degree, ord; body = body_order)

        if plotting == true
            # check orthogonality
            heatmap(A_pure[:,1:length(NN2b)]' * A_pure[:,1:length(NN2b)] , aspect_ratio=1)
            savefig(exp_dir * "check_A_pure2b")
            if length(NN3b) > 0
                heatmap(A_pure[:,length(NN2b) + 1:end]' * A_pure[:,length(NN2b) + 1:end] , aspect_ratio=1)
                savefig(exp_dir * "check_A_pure3b")
            end
        end

        sol_pure = solveLSQ(A_pure, Y; solver=solver)
        
        # testing
        XX_test = rand(distribution(-1, 1), (num_test, K_R))
        A_test_E = designMatNB(XX_test, poly, max_degree, ord; body = body_order)

        zs_pure = A_test_E * sol_pure
        ground_zs = Testing_func(XX_test)
        curr_err_E = norm(zs_pure - ground_zs)/norm(ground_zs)

        println("relative error of E_avg: ", curr_err_E)
        pure_err_curr += curr_err_E

        # get f error
        XX_test = range(-1, 1, length=testSampleSize)

        A_test = predMatNB(XX_test, poly, max_degree, ord; body = body_order)
        yp = A_test * sol_pure
        ground_yp = f.(XX_test)
        RMSE = norm(yp - ground_yp)/num_sam
        curr_err_f = norm(yp - ground_yp)/norm(ground_yp)

        println("Relative error of f: ", curr_err_f)
        println("RMSE of f: ", RMSE)

        err_f_curr += curr_err_f

        if length(NN2b) > 0
            cond_num_2b_pure_curr += cond(A_pure[:, 1:max_degree])
            @show cond_num_2b_pure_curr
        end

        if length(NN3b) > 0
            cond_num_3b_pure_curr += cond(A_pure[:, length(NN2b)+1: end])
            @show cond_num_3b_pure_curr
        end
    end

    push!(pure_err, pure_err_curr / 10)
    push!(err_f, err_f_curr/10)
    push!(cond_num_2b_pure, cond_num_2b_pure_curr / 10) 
    push!(cond_num_3b_pure, cond_num_3b_pure_curr/ 10)
end

# -------------------------------------------------------result saving------------------------------------------#

if plotting == true
    p1 = plot(num_sam_list, pure_err, xscale=:log10, label = "Eavg", xlabel = "num sam", ylabel = "rel err", yscale = :log10)
    plot!(p1, num_sam_list, err_f, xscale=:log10, label = "V", xlabel = "num sam", ylabel = "rel err", yscale = :log10)
    savefig(p1, exp_dir*"/checkerror_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]")

    p2 = plot(num_sam_list, cond_num_2b_pure, xscale=:log10, label = "2b", xlabel = "num sam", ylabel = "Coniditional number")
    savefig(p2, exp_dir*"/checkcond_2b_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]")

    p4 = plot(num_sam_list, cond_num_3b_pure, xscale=:log10, label = "3b", xlabel = "num sam", ylabel = "Coniditional number")
    savefig(p4, exp_dir*"/checkcond_3b_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]")

    CSV.write(exp_dir*"/err_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]" * ".csv", (pure_err = pure_err)) 
    CSV.write(exp_dir*"/cond_sep_numsam=[" * string(aa) * "," * string(bb) * "]" * "_maxdeg=[" * string(max_degree) * "]" * ".csv", (cond_num_2b_pure = cond_num_2b_pure, cond_num_3b_pure = cond_num_3b_pure)) 
end

 

