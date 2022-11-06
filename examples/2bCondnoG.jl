using LinearAlgebra, Polynomials4ML
# function legendre(x::Number, N)
#     L = zeros(N+1)
#     L[1] = 1
#     L[2] = x
#     for n = 2:N
#         L[n+1] = (2*n-1)/n * x * L[n] - (n-1)/n * L[n-1]
#     end
#     for n = 0:N
#         L[n+1] *= 2*n+1
#     end
#     return L
#  end
 
#  function cheb(x::Number, N)
#     T = zeros(N+1)
#     T[1] = 1 
#     T[2] = x 
#     for n = 2:N 
#        T[n+1] = 2*x*T[n] - T[n-1]
#     end 
#     return T 
# end
cheb(N) = chebyshev_basis(N)
legendre(N) = legendre_basis(N, normalize=true) 

r_cut = 1
r_in = -1
# r_cut = 3
# r_in = 0.85
K_R = 4
p = 4
env(r) = (r^(-p) - r_cut^(-p) + p * r_cut^(-p-1) * (r - r_cut)) * (r < r_cut)


function design_matrix_env(xs, basis, N)
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

let N = 10, M=10, K_R = 1
    Rs = [rand(K_R)*(r_cut-r_in) .+ r_in for _=1:M]
    A = design_matrix(Rs, legendre, N)
    @show cond(A'*A)
end

let Ns = [10, 20], Ms = [20, 100], K_Rs = [1, 2, 4, 32], basis_list = [cheb, legendre]
    for basis in basis_list
        for N in Ns
            for M in Ms
                for K_R in K_Rs
                    Rs = [rand(K_R)*(r_cut-r_in) .+ r_in for _=1:M]
                    A = design_matrix(Rs, basis, N)
                    A_env = design_matrix_env(Rs, basis, N)
                    @show basis, N, M, K_R
                    @show cond(A'*A)
                    @show cond(A_env'*A_env)
                end
            end
        end
    end
end


# simple case
N = 10
Xs = LinRange(-1, 1, N)
Data = cheb(N)(Xs)
@show size(Data)

# @show 1/N * Data' * Data

Data = legendre(N)(Xs)
@show size(Data)

G = Data' * Data
@show 1/N * G
@show cond(G)

# for pp = 1:max_degree
#     for qq = 1:max_degree
#         int = dot(Data[:, pp], Data[:, qq]) / N
#         @show int
#         if abs(int) > 1e-7
#             @show pp, qq
#         end
#     end
# end

# function simpson(f, a, b, N)
#     h = (b-a)/N 
#     q = 0.0 
#     for n = 0:N-1
#         x0 = a + h*n 
#         x1 = x0 + h/2 
#         x2 = x0 + h 
#         q += (h/6) * (f(x0) + 4 * f(x1) + f(x2))
#     end
#     return q 
# end

