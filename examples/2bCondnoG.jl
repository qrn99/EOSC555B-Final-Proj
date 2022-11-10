include("../src/utils.jl")
using LinearAlgebra, Polynomials4ML
##
cheb(N) = chebyshev_basis(N)
legendre(N) = legendre_basis(N, normalize=true) 
##

# r_cut = 1
# r_in = -1

# K_R = 4

r_in = 0.85 # with envelope should be 0.0
r_nn = 1.0
r_cut = 3.0

# morse transform
# x1(r) = 1 / (1 + r / r_nn)

#Agnesis Transform
# 0.33 = a = (p-1)/(p+1)
x1(r) = 1.0 / (1.0 + 0.33*(r / r_nn)^2)

x_in = x1(0.0); x_cut = x1(r_cut) # Regularize til r=0
x(r) = 2 * (x1(r) - x_cut) / (x_in - x_cut) - 1

p = 4
env(r) = (r^(-p) - r_cut^(-p) + p * r_cut^(-p-1) * (r - r_cut)) * (r < r_cut)


function design_matrix(xs, basis, N)
    A = zeros(ComplexF64, length(xs), N)
    for (i, xx) in enumerate(xs)
      A[i, :] = sum(evaluate(basis(N), x) for x in xx) / length(xx)
    end 
    return A
end

function design_matrix_env(xs, basis, N)
    A = zeros(ComplexF64, length(xs), N)
    for (i, xx) in enumerate(xs)
      A[i, :] = sum(evaluate(basis(N), x) * env(x) for x in xx) / length(xx)
    end 
    return A
end

function design_matrix_cordtrans_env(rs, basis, N)
    A = zeros(ComplexF64, length(rs), N)
    for (i, rr) in enumerate(rs)
      A[i, :] = sum(evaluate(basis(N), x(r)) * env(r) for r in rr) / length(rr)
    end 
    return A
end

let N = 10, M=1000, K_R = 1
    Rs = [rand(K_R)*(r_cut-r_in) .+ r_in for _=1:M]
    A = design_matrix(Rs, legendre, N)
    @show cond(A'*A)
end

## 
# averaged 2B
let Ns = [10, 20], Ms = [20, 100, 1000], K_Rs = [1, 2, 4], basis_list = [cheb, legendre]
    for basis in basis_list
        for N in Ns
            for M in Ms
                for K_R in K_Rs
                    Rs = [rand(K_R)*(r_cut-r_in) .+ r_in for _=1:M]
                    A = design_matrix(Rs, basis, N)
                    A_env = design_matrix_env(Rs, basis, N)
                    A_env_ct = design_matrix_cordtrans_env(Rs, basis, N)
                    G_norm = (M*norm(A, 2))^2
                    @show basis, N, M, K_R
                    @show G_norm
                    @show cond(A'*A)
                    @show cond(A_env'*A_env)
                    @show cond(A_env_ct'*A_env_ct)
                end
            end
        end
    end
end
##

##
# For the A^TA LSQ stability of averaged energy design matrix A  
# Check unknown prodcut term behaviour as M → ∞
let Ns = [20], Ms = [20, 100, 1000], K_Rs = [1, 2, 4], basis_list = [legendre]
    r_cut = 1
    r_in = -1
    for basis in basis_list
        for N in Ns
            for K_R in K_Rs
                for M in Ms
                    Rs = [rand(K_R)*2 .- 1 for _=1:M]
                    sum1_mat = zeros(N, N)
                    sum2_mat = zeros(N, N)
                    for n=1:N
                        for n2=1:N
                            sum1 = 0
                            sum2 = 0
                            for (i, rr) in enumerate(Rs)
                                for k=eachindex(rr)
                                    for k2=eachindex(rr)
                                        if k == k2
                                            # @show evaluate(cheb(N), rr[k])
                                            sum1 += evaluate(cheb(N), rr[k])[n] * evaluate(cheb(N), rr[k'])[n2]
                                        else
                                            sum2 += evaluate(cheb(N), rr[k])[n] * evaluate(cheb(N), rr[k'])[n2]
                                        end
                                    end
                                end
                            end
                            sum1_mat[n, n2] = sum1
                            sum2_mat[n, n2] = sum2
                        end
                    end
                    @show (M, N, K_R)
                    heatmap(sum1_mat/(K_R^2), aspect_ratio=1)
                    heatmap(sum2_mat/(K_R^2), aspect_ratio=1)
                end
            end
        end
    end
end
##

##
function getSumDiffProduct(Rs, basis, N, K_R)
    basis_eval = basis(N)
    sum1_mat = zeros(N, N)
    sum2_mat = zeros(N, N)
    for (i, rr) in enumerate(Rs)
        sum1_i = zeros(N, N)
        sum2_i = zeros(N, N)
        for k=eachindex(rr)
            for k2=eachindex(rr)
                if k == k2
                    # outer product
                    sum1_i += evaluate(basis_eval, rr[k]) .* evaluate(basis_eval, rr[k'])'
                else
                    sum2_i += evaluate(basis_eval, rr[k]) .* evaluate(basis_eval, rr[k'])'
                end
            end
        end
        sum1_mat += sum1_i 
        sum2_mat += sum2_i
    end
    return sum1_mat/(K_R^2), sum2_mat/(K_R^2)
end

# function des_mat2(Rs, basis, N, K_R)
#     basis_eval = basis(N)
#     sum2 = zeros(N, N)
#     for (i, rr) in enumerate(Rs)
#         @show sum(evaluate(basis_eval, r) for r in rr)
#         sum2 += sum(evaluate(basis_eval, r) for r in rr) * sum(evaluate(basis_eval, r) for r in rr)'
#     end
#     return sum2/(K_R^2)
# end
# A2 = des_mat2(Rs, legendre, N, K_R)

function design_matrix_no_scale(xs, basis, N)
    A = zeros(ComplexF64, length(xs), N)
    for (i, xx) in enumerate(xs)
      A[i, :] = sum(evaluate(basis(N), x) for x in xx)/ length(xx)
    end
    return A
end

N = 10
M = 100
K_R = 2
Rs = [rand(K_R)*2 .- 1 for _=1:M]
sum1, sum2 = getSumDiffProduct(Rs, legendre, N, K_R)
A = real(design_matrix(Rs, legendre, N))
# A2 = des_mat2(Rs, legendre, N, K_R)
# A'*A .== A2
A'*A
A'*A .== sum1+sum2
@assert(A'*A == sum1+sum2)
A'*A
sum1+sum2
@show (sum1+sum2)
heatmap(sum1 , aspect_ratio=1)
heatmap(sum2, aspect_ratio=1)
heatmap(A'*A, aspect_ratio=1)
##


##
# simple 2B
N = 10
M = 100
# Xs = LinRange(-1, 1, M)
Xs = rand(N)*2 .- 1
# Data = cheb(N)(Xs)
# @show size(Data)

# @show 1/N * Data' * Data

Data = legendre(N)(Xs)
@show size(Data)

G = Data' * Data
@show 1/N * G
@show cond(G)

let Ns = [10, 20], Ms = [20, 100, 1000], basis_list = [cheb, legendre]
    for basis in basis_list
        for N in Ns
            for M in Ms
                # Xs = LinRange(-1, 1, M)
                Xs = rand(M).*2 .-1
                Data = basis(N)(Xs)
                G = Data' * Data
                @show basis, N, M
                @show cond(G)
            end
        end
    end
end
#%%

# simple 3B
N = 1000 #num of samples
max_degree = 15
X1 = LinRange(-1, 1, N)
X1 = collect(LinRange(-1, 1, N))

# X1 = rand(N) .* 2 .- 1
X2 = LinRange(-1, 1, N)
X2 = collect(LinRange(-1, 1, N))
# X2 = rand(N) .* 2 .- 1
NN = get_NN(max_degree)

function G_innerProd_3b(X1, X2, poly, NN2)
    D1 = poly(X1)
    D2 = poly(X2)

    B = length(NN2)
    G = zeros((length(X1), B))
    for b=1:B
        n, m = NN2[b]
        G[:, b] = D1[:, n] .* D2[:, m] + D1[:, m] .* D2[:, n]
    end

    return G
end

poly = legendre_basis(max_degree, normalize = true)
G = G_innerProd_3b(X1, X2, poly, NN[max_degree + 1:end])
@show cond(G)
@show "||| M ||| ",  sqrt(1/N * norm(G, 2))

##

