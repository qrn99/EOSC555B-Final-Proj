module HelperFunctions


using Plots, LinearAlgebra, Random, Statistics, Distributions, Polynomials4ML
using PyCall, JuLIP

export cheb, legendre, design_matrix, get_basis, generate_data, generate_data_dst,
        lr, predict, LSQ_error, LSQ_error_deg,
        brr, predict_bayes, brr_error, brr_error_deg,
        prior_default, prior_H, prior_A, 
        lsq_sk


# TODO: add poly

# polynomail construction of degree N, output is [P₀, ..., Pₙ]
cheb(N) = chebyshev_basis(N)
legendre(N) = legendre_basis(N, normalize=true) 

"""
    spatial2rad(xs)
    Map (x, y) → (r, θ) in 2D

    * @param xs          Vector{Vector(Float64)}          angular info of xs
    * @return xs_re      Vector{Vector(Float64)}                  exp basis matrix
"""
function spatial2rad(xs)
    num_sam, K_R, _ = size(xs)
    xs_re = zeros(size(xs))
    for i = 1:num_sam        
        for j = 1:K_R
            xs_re[i,j,1], xs_re[i,j,2] = sqrt(xs[i, j, 1]^2 + xs[i, j, 2]^2), atan(xs[i, j, 2]/xs[i, j, 1])
        end
    end
    return xs_re
end

"""
    exp_basis(xs_theta, deg)
    Returns the basis matrix for expotential basis
    
    * @param xs_theta          Vector{Vector(Float64)}          angular info of xs
    * @param deg               Int                              max deg of exp function
    * @return B             Matrix{ComplexF64}                  exp basis matrix
"""
function exp_basis(xs_theta, deg)
    num_sam = length(xs_theta)
    B = zeros(ComplexF64, num_sam, deg)
    for k = 1:num_sam
        for j = 1:deg
            B[k, j] = exp(1im * j * xs_theta[k])
        end
    end
    return B
end


"""
    designMatNB2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody)
    Returns the design matrix for averaged estimation 2D model and spec
    
    * @param train          Vector{Vector{Vector(Float64)}}      set of atomic strucutres (1d) is array of array of radius distances
    * @param poly_basis     OrthPolyBasis1D3T{Float64}          basis function
    * @param max_deg_poly   Int64                               maximum degree of poly basis function
    * @param max_deg_exp    Int64                               maximum degree of exp basis function
    * @param body           Int64                               body order
    * @return               Matrix{ComplexF64}                  design matrix
    * @return               Vector{tuple(Int)}                  specification of the design matrix

    # Examples
    ```jldoctest
    num_sam = 10
    K_R = 4
    train = rand(num_sam, K_R, 2)
    poly_basis = chebyshev_basis(10)
    max_deg_poly = 5
    max_deg_exp = 5
    ord = 2
    A, spec = designMatNB2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody)
    ```
"""
function designMatNB2D(train, poly_basis, max_deg_poly, max_deg_exp, ord; body=:TwoBodyThreeBody) # TODO: remove ord later

    NN = get_NN(max_deg_poly, ord)
    NN2b = NN[length.(NN) .== 1]
    NN3b = NN[length.(NN) .== 2]
    M, K_R, _ = size(train)
    xs_rad = spatial2rad(train) # num_data × K_R × 2
    poly_list = [poly_basis(xs_rad[:, i, 1]) for i = 1:K_R] #  K_R × num_data × length(poly_basis) 
    exp_list = [exp_basis(xs_rad[:, i, 2], max_deg_exp) for i = 1:K_R] #  K_R × num_data × max_deg_exp
    spec = []
    if body == :TwoBody # 2body interaction
        A = zeros(ComplexF64, M, length(NN2b) * max_deg_exp)
        for i = 1:length(NN2b)
            nn = NN2b[i]
            @show nn
            for j = 1:max_deg_exp
                A[:, (i-1) * max_deg_exp + j] = sum([PX1[:, nn] .* EX1[:, j]  for PX1 in poly_list for EX1 in exp_list])
                push!(spec, [(nn[1], j)])
            end
        end
    elseif body == :ThreeBody #3body interaction
        A = zeros(ComplexF64, M, length(NN3b) * max_deg_exp  * max_deg_exp)
        for i = 1:length(NN3b)
            nn, mm = NN3b[i]
            for j = 1:max_deg_exp
                for p = 1:max_deg_exp
                    if j + p <= max_deg_exp
                        A[:, (i-1) * max_deg_exp * max_deg_exp + (j-1)*max_deg_exp + p] = sum([PX1[:, nn] .* PX2[:, mm] .* EX1[:, j] .* EX2[:, p] for PX1 in poly_list for PX2 in poly_list for EX1 in exp_list for EX2 in exp_list if (PX1 != PX2 && EX1 != EX2)])
                        push!(spec, [(nn, j), (mm, p)])
                    end
                end
            end
        end
    elseif body == :TwoBodyThreeBody #both 2b3b
        A = zeros(ComplexF64, M, length(NN) * max_deg_exp * max_deg_exp)
        for i = 1:length(NN2b)
            nn = NN2b[i]
            for j = 1:max_deg_exp
                A[:, (i-1) * max_deg_exp + j] = sum([PX1[:, nn] .* EX1[:, j]  for PX1 in poly_list for EX1 in exp_list])
                push!(spec, [(nn[1], j)])
            end
        end
        for i = 1:length(NN3b)
            nn, mm = NN3b[i]
            for j = 1:max_deg_exp
                for p = 1:max_deg_exp
                    if j + p <= max_deg_exp
                        A[:, length(NN2b) * max_deg_exp + 1 + (i-1) * max_deg_exp * max_deg_exp + (j-1)*max_deg_exp + p] = sum([PX1[:, nn] .* PX2[:, mm] .* EX1[:, j] .* EX2[:, p] for PX1 in poly_list for PX2 in poly_list for EX1 in exp_list for EX2 in exp_list if (PX1 != PX2 && EX1 != EX2)])
                        push!(spec, [(nn, j), (mm, p)])
                    end
                end
            end
        end
    else
        println("Does not support this body order.")
    end
    return real(A), spec
end

"""
    design_matrix(xs, basis, N)
    Returns the design matrix for averaged estimation model
    
    * @param xs      Vector{Vector(Float64)}      set of atomic strucutres (1d) is array of array of radius distances
    * @param basis   OrthPolyBasis1D3T{Float64}   basis function
    * @param N       Int64                        maximum degree of basis function
    * @return        Matrix{ComplexF64}           [length(xs) x N] design matrix
    
    # Examples
    ```jldoctest
    julia> design_matrix([[1.0, 2.0]], chebyshev_basis(5), 5)
    1×5 Matrix{ComplexF64}:
    0.56419+0.0im  1.19683+0.0im  3.19154+0.0im  10.7714+0.0im  39.0963+0.0im
    ```
"""
function design_matrix(xs, basis, N)
    A = zeros(ComplexF64, length(xs), N)
    for (i, xx) in enumerate(xs)
      A[i, :] = sum(evaluate(basis, x) for x in xx) / length(xx)
    end 
    return A
end

"""
    lr(averge_est_fnc, target_fnc, train, basis, N; noise=0.0)
    Returns the optimial parameter of the least square problem of the linear polynomial regression model
    
    * @param averge_est_fnc     generic fnc                     average estimate function
    * @param target_fnc         generic fnc                     target function
    * @param train              Vector{Vector(Float64)}         set of atomic strucutres (1d) is array of array of radius distances
    * @param basis              OrthPolyBasis1D3T{Float64}      basis function
    * @param N                  Int64                           maximum degree of basis function
    * @keyargs noise            Float64                         [default 0] noise in data
    * @return                   Vector{ComplexF64}              [N-element] optimial parameter of the lr model
    
    # Examples
    ```jldoctest
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> lr(E, f1, [[1.0, 2.0]], chebyshev_basis(5), 5)
    5-element Vector{ComplexF64}:
    0.001532675294821725 + 0.0im
    0.003251295295862509 + 0.0im
    0.008670120788966687 + 0.0im
    0.029261657662762557 + 0.0im
    0.10620897966484187 + 0.0im
    ```
"""
function lr(averge_est_fnc, target_fnc, train, basis, N; noise=0.0)
    Y = averge_est_fnc(target_fnc, train) .+ noise
    Ψ = design_matrix(train, basis, N)
    Q,R = qr(Ψ)
    #implement QR factorization
    μ = R \ (Matrix(Q)'*Y)
    # μ = Ψ \ Y 
    return μ
end

"""
    predict(r::Float64, basis, μ)
    Returns the predicted value of linear polynomial regression model with parameter μ at a prediction node r
    
    * @param r          Float64                         prediction node
    * @param basis      OrthPolyBasis1D3T{Float64}      basis function
    * @param μ          Vector{ComplexF64}              [N-element] optimial parameter of the lr model
    * @return           Float64                         predicted value of lr model at the prediction node
    
    # Examples
    ```jldoctest
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> basis = chebyshev_basis(5)
    julia> μ = lr(E, f1, [[1.0, 2.0]], basis, length(basis))
    julia> predict(1.5, basis, μ)
    2.230543812448502
    ```
"""
function predict(r::Float64, basis, μ)
   B = evaluate(basis, r)
   val = real(sum(μ .* B))
   return val
end

"""
    @overload
    predict(rr, args...)
    Returns the predicted values of linear polynomial regression model with parameter μ at a list of prediction nodes rr

    * @param rr         Vector{Float64}                 list of prediction nodes
    * @args...
        * @param basis  OrthPolyBasis1D3T{Float64}      basis function
        * @param μ      Vector{ComplexF64}              [N-element] optimial parameter of the lr model
    * @return           Vector{Float64}                 list of predicted value of lr model at the prediction nodes

    # Examples
    ```jldoctest
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> basis = chebyshev_basis(5)
    julia> μ = lr(E, f1, [[1.0, 2.0]], basis, length(basis))
    julia> predict([1.5, 2.5], basis, μ)
    2-element Vector{Float64}:
     2.230543812448502
    23.700662951604944
    ```
"""
function predict(rr, args...)
   vals = Float64[]
   for r in rr
      v = predict(r, args...)
      push!(vals, v)
   end
   return vals
end

"""
    generate_data(M, K_R, testSampleSize, data_dst, dst; r_in=-1, r_cut=1)
    Returns a pair of train and test sample dataset given the data distribtuion

    * @param M                  Int64       number of atomic strucutres in the train dataset
    * @param K_R                Int64       number of radius distances in one atomic structure
    * @param testSampleSize     Int64       size of test dataset
    * @param data_dst           Dict{String, Distributions.Censored{}}      a dict of data distribtuions
    * @param dst                String      chosen data distritbuion name
    * @keyargs r_in             Float64     [default -1] lower bound of data
    * @keyargs r_cut            Float64     [default 1] upper bound of data
    * @return                   { Vector{Vector{Float64}}, 
                                Vector{Vector{Float64}} }                   train and test sample dataset
    
    # Examples
    ```jldoctest
    julia> data_dst = generate_data_dst(0.3, -1, 1, -1, 1)
    julia> dst = "uniform"
    julia> train, test = generate_data(10, 2, 2, data_dst, dst)
    julia> test
    2-element Vector{Vector{Float64}}:
     [0.6870254428602236, -0.9871534302299474]
     [-0.7419329796675329, 0.47381413887549306]
    ```
"""
function generate_data(M, K_R, testSampleSize, data_dst, dst; r_in=-1, r_cut=1)
    Random.seed!(12216859)
    # generate data based on which data distribution is wanted
    # TODO: change test to more points in same distribution? Check how to plot model prediction...
    if dst == "uniform"
        train = [rand(K_R)*(r_cut-r_in) .+ r_in for _=1:M]
        test = [rand(K_R)*(r_cut-r_in) .+ r_in for _=1:testSampleSize]
    elseif dst == "cheb"
        train = [cos.(π*rand(K_R)) for _=1:M]
        test = [cos.(π*rand(K_R)) for _=1:testSampleSize]
    else
        dst_prob = data_dst[dst]
        train = [rand(dst_prob, K_R) for _=1:M]
        test = [rand(dst_prob, K_R) for _=1:testSampleSize]
    end

    return train, test
end

"""
    generate_data_dst(rdf_bimodal, r_in, r_cut, left, right; abs_width=1, hole_width=1/2)
    Returns a dict of data distribtuions that has bimodal, absolute, and hole properties

    * @param rdf_bimodal    Float64     how wide each peak is for the bimodal distribtuion
    * @param r_in           Float64     the centered of left peak
    * @param r_cut          Float64     the centered of right peak
    * @param left           Float64     the lower cutoff bound of the distribution
    * @param right          Float64     the upper cutoff bound of the distribution
    * @keyargs abs_width    Float64     [default 1] how wide the absolute distribtuion has being a union of two side of triangles
    * @keyargs hole_width   Float64     [default 0.5] how close the hole distribution has being a union of two side of triangles
    * @return               Dict{String, Distributions.Censored{}}        dict of 3 data distribtuions that has bimodal, absolute, and hole properties
    
    # Examples
    ```jldoctest
    julia> generate_data_dst(0.3, -1, 1, -1, 1)
    Dict{String, Distributions.Censored{D, Continuous, Int64, Int64, Int64} where D<:(UnivariateDistribution)} with 3 entries:
    "hole"    => Distributions.Censored{MixtureModel{Univariate, Continuous, SymTriangularDist{Float64}, Categorical{Float64, Vector{Float64}}}, Continuous, Int64, Int64, In…
    "bimodal" => Distributions.Censored{MixtureModel{Univariate, Continuous, Distribution{Univariate, Continuous}, Categorical{Float64, Vector{Float64}}}, Continuous, Int64,…
    "abs"     => Distributions.Censored{MixtureModel{Univariate, Continuous, SymTriangularDist{Float64}, Categorical{Float64, Vector{Float64}}}, Continuous, Int64, Int64, In…
    ```
"""
function generate_data_dst(rdf_bimodal, r_in, r_cut, left, right; abs_width=1, hole_width=1/2)
    bimodal = censored(MixtureModel([Normal(r_in, rdf_bimodal), 
                Normal(r_cut, rdf_bimodal), Uniform(r_in, r_cut)], [0.35, 0.35, 0.3]), left, right)
    abs = censored(MixtureModel([SymTriangularDist(r_in, abs_width), SymTriangularDist(r_cut, abs_width)]), left, right)
    hole = censored(MixtureModel([SymTriangularDist(r_in, hole_width), SymTriangularDist(r_cut, hole_width)]), left, right)
    
    data_dst = Dict("bimodal" => bimodal, "abs" => abs, "hole" => hole)
    
    return data_dst
end

"""
    get_basis(basis_choice, N, K_R, data_dst, dst; adaptedTrainSize=100, testSampleSize=100)
    Returns the instantiated basis function

    * @param basis_choice   String/function   choice of basis
    * @param N              Int64     the maximum degree of basis
    * @param K_R            Int64     the number of radius distances per structure
    * @param data_dst       Dict{String, Distributions.Censored{}}      a dict of data distribtuions
    * @param dst            String    chosen data distritbuion name
    * @keyargs adaptedTrainSize     Int64     [default 100] number of train data to train the adapted basis
    * @keyargs testSampleSize       Int64     [default 100] number of test data
    * @return               OrthPolyBasis1D3T{Float64}      basis function               
    
    # Examples
    ```jldoctest
    julia> data_dst = generate_data_dst(0.3, -1, 1, -1, 1)
    julia> dst = "uniform"
    julia> get_basis("OrthPoly", 5, 2, data_dst, dst)
    OrthPolyBasis1D3T{Float64}(..., Dict{String, Any}("weights" => DiscreteWeights{Float64}(...)
    ```
"""
function get_basis(basis_choice, N, K_R, data_dst, dst; adaptedTrainSize=100, testSampleSize=100)
    if basis_choice == "OrthPoly"
        train_orth, _ = generate_data(adaptedTrainSize, K_R, testSampleSize, data_dst, dst)
        rdf = vcat(train_orth...)
        W = DiscreteWeights(vcat(rdf), ones(length(rdf)), :normalize)
        basis = orthpolybasis(N, W)
    else 
        basis = basis_choice(N)
    end
    return basis
end

"""
    LSQ_error(NN, MM, K_R, testSampleSize, target_fn, averge_est_fn, basis, data_dst, dst; noise=0.0, test_uniform=false)
    Returns a list of prediction error for the linear polynomail regression model given a list of degree of basis NN and a list of number of samples MM

    * @param NN                 Vector{Int64}   list of degree of basis
    * @param MM                 Vector{Int64}   list of number of samples
    * @param K_R                Int64           the number of radius distances per structure in the sample data
    * @param testSampleSize     Int64           the number of test data
    * @param target_fnc         generic fnc     target function
    * @param averge_est_fnc     generic fnc     average estimate function
    * @param basis              OrthPolyBasis1D3T{Float64}                  basis function        
    * @param data_dst           Dict{String, Distributions.Censored{}}      a dict of data distribtuions
    * @param dst                String          chosen data distritbuion name for the train data
    * @keyargs noise            Float64         [default 0] noise in data
    * @keyargs test_uniform     Bool            [default false] whether the test data is genereated from unifrom distribution
    * @return                   Matrix{Float64} [1 × length(MM)] list of prediction error (RSME)       
    
    # Examples
    ```jldoctest
    julia> MM=[40, 80, 160]; NN = [20 for _ in 1:length(MM)]; K_R = 2
    julia> data_dst = generate_data_dst(0.3, -1, 1, -1, 1)
    julia> dst = "uniform"
    julia> basis = get_basis("OrthPoly", NN[1], K_R, data_dst, dst)
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> LSQ_error(NN, MM, K_R, 100, f1, E, basis, data_dst, dst)
    1×3 Matrix{Float64}:
     6.91015e-5  7.14561e-5  6.07153e-5
    ```
"""
function LSQ_error(NN, MM, K_R, testSampleSize, target_fn, averge_est_fn, basis, data_dst, dst; noise=0.0, test_uniform=false)
    err = zeros(1, length(NN))

    for i=1:length(NN)
        N = NN[i]
        M = MM[i]

        train, test = generate_data(M, K_R, testSampleSize, data_dst, dst)

        #get parameter
        μ = lr(averge_est_fn, target_fn, train, basis, N; noise=noise)

        if test_uniform
            test = sort(collect(Iterators.flatten([ rand(K_R)*2 .- 1 for _=1:testSampleSize ])))
        else
            test = sort(collect(Iterators.flatten(test)))
        end
        
        yp = target_fn.(test)
        y = predict(test, basis, μ)

        err[i] = norm(yp - y, 2)/sqrt(length(test))
    end
    
    return err
end

"""
    LSQ_error(NN, MM, K_R, testSampleSize, target_fn, averge_est_fn, basis, data_dst, dst; noise=0.0, test_uniform=false)
    Returns a list of prediction error for the linear polynomail regression model given a list of degree of basis NN and a list of number of samples MM

    * @param NN                 Vector{Int64}   list of degree of basis
    * @param MM                 Vector{Int64}   list of number of samples
    * @param K_R                Int64           the number of radius distances per structure in the sample data
    * @param testSampleSize     Int64           the number of test data
    * @param target_fnc         generic fnc     target function
    * @param averge_est_fnc     generic fnc     average estimate function
    * @param basis_choice       String/function choice of basis
    * @param data_dst           Dict{String, Distributions.Censored{}}      a dict of data distribtuions
    * @param dst                String          chosen data distritbuion name for the train data
    * @keyargs noise            Float64         [default 0] noise in data
    * @keyargs test_uniform     Bool            [default false] whether the test data is genereated from unifrom distribution
    * @keyargs adaptedTrainSize Int64           [default 100] number of train data to train the adapted basis
    * @return                   Matrix{Float64} [1 × length(MM)] list of prediction error (RSME)       
    
    # Examples
    ```jldoctest
    julia> NN = [20, 40, 60]; MM= NN.^2; K_R = 2
    julia> data_dst = generate_data_dst(0.3, -1, 1, -1, 1)
    julia> dst = "uniform"
    julia> basis_choice = "OrthPoly"
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> LSQ_error_deg(NN, MM, K_R, 100, f1, E, basis_choice, data_dst, dst)
    1×3 Matrix{Float64}:
     6.45463e-5  4.49789e-6  9.70116e-7
    ```
"""
function LSQ_error_deg(NN, MM, K_R, testSampleSize, target_fn, averge_est_fn, basis_choice, data_dst, dst; noise=0.0, test_uniform=false, adaptedTrainSize=100)
    err = zeros(1, length(NN))

    for i=1:length(NN)
        N = NN[i]
        M = MM[i]
        
        train, test = generate_data(M, K_R, testSampleSize, data_dst, dst)
        
        basis = get_basis(basis_choice, N, K_R, data_dst, dst; adaptedTrainSize)

        #get parameter
        μ = lr(averge_est_fn, target_fn, train, basis, N; noise=noise)

        if test_uniform
            test = sort(collect(Iterators.flatten([ rand(K_R)*2 .- 1 for _=1:testSampleSize ])))
        else
            test = sort(collect(Iterators.flatten(test)))
        end
        
        yp = target_fn.(test)
        y = predict(test, basis, μ)

        err[i] = norm(yp - y, 2)/sqrt(length(test))
    end
    
    return err
end

### Bayesian version ###

# no reweighting
prior_default(N, α) = 1/α * I(N) 

# Algebraic/Hilbert reweighting
prior_H(N, j, α) = Diagonal( [ (1+n^j)/α for n=0:(N-1) ] )

# Analytical reweighting
prior_A(N, ρ, α) = Diagonal( [ (ρ^n)/α for n=0:(N-1) ] )

"""
    brr(averge_est_fnc, target_fnc, train, basis, N; noise=0.0, prior="Identity", β_inv=1e-6, ρ=1, j=1, α=1.0)
    Returns the posterior parameters (mean, covariance) of the bayesian ridge regression model
    
    * @param averge_est_fnc     generic fnc                     average estimate function
    * @param target_fnc         generic fnc                     target function
    * @param train              Vector{Vector(Float64)}         set of atomic strucutres (1d) is array of array of radius distances
    * @param basis              OrthPolyBasis1D3T{Float64}      basis function
    * @param N                  Int64                           maximum degree of basis function
    * @keyargs noise    Float64     [default 0] noise to perturbs the data
    * @keyargs prior    String      [default "Identity" or "Hilbert", "Analy"] bayesain parameter smoothness prior assumption
    * @keyargs β_inv    Float64     [default 1e-6] precision parameter
    * @keyargs ρ        Int64       [default 1] analytical smoothness strength
    * @keyargs j        Int64       [default 1] algebraic smoothness strength
    * @keyargs α        Float64     [default 1.0] regularization coefficient
    * @return           {Vector{ComplexF64}, Matrix{ComplexF64}}    [N, N × N] posterior parameters (mean, covariance) of the bayesian ridge regression model
    
    # Examples
    ```jldoctest
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> μ, Σ_inv = brr(E, f1, [[1.0, 2.0]], chebyshev_basis(5), 5)
    julia> μ
    5-element Vector{ComplexF64}:
     0.0015326753348992963 + 0.0im
     0.003251295328333746 + 0.0im
     0.00867012168231078 + 0.0im
     0.029261657001422556 + 0.0im
     0.10620897970306474 + 0.0im
    julia> Σ_inv
    5×5 Matrix{ComplexF64}:
     3.18311e5+0.0im  6.75237e5+0.0im  1.80063e6+0.0im  6.07714e6+0.0im  2.20578e7+0.0im
     6.75237e5-0.0im   1.4324e6+0.0im  3.81972e6+0.0im  1.28916e7+0.0im  4.67916e7+0.0im
     1.80063e6-0.0im  3.81972e6-0.0im  1.01859e7+0.0im  3.43775e7+0.0im  1.24777e8+0.0im
     6.07714e6-0.0im  1.28916e7-0.0im  3.43775e7-0.0im  1.16024e8+0.0im  4.21124e8+0.0im
     2.20578e7-0.0im  4.67916e7-0.0im  1.24777e8-0.0im  4.21124e8-0.0im  1.52852e9+0.0im
    ```
"""
function brr(averge_est_fnc, target_fnc, train, basis, N; noise=0.0, prior="Identity", β_inv=1e-6, ρ=1, j=1, α=1.0)
    Y = averge_est_fnc(target_fnc, train) .+ noise
    
    Ψ = design_matrix(train, basis, N)
    
    # prior
    if prior == "Hilbert"
       A = prior_H(N, j, α)
    elseif prior == "Analy"
       A = prior_A(N, ρ, α)
    else
       A = prior_default(N, α)
    end

    # posterior
    Σ_inv = A + (1/β_inv) * Ψ' * Ψ
    μ = (1/β_inv) * ( Σ_inv \ (Ψ' * Y) )
    return μ, Σ_inv
end

"""
    predict_bayes(r::Float64, basis, μ, Σ_inv, β_inv)
    Returns the predicted value of bayesian ridge regression model with parameter μ, Σ_inv, precision β, at a prediction node r
    
    * @param r          Float64                         prediction node
    * @param basis      OrthPolyBasis1D3T{Float64}      basis function
    * @param μ          Vector{ComplexF64}              [N-element] posterior distribution's average parameter of the brr model
    * @param Σ_inv      Matrix{ComplexF64}              [N × N] posterior distribution's average covariance matrix of the brr model
    * @param β_inv      Float64                         precision parameter of the brr model
    * @return           {Float64, Float64}              predicted value of brr model with bayesain uncertainty at the prediction node
    
    # Examples
    ```jldoctest
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> basis = chebyshev_basis(5)
    julia> μ, Σ_inv = brr(E, f1, [[1.0, 2.0]], basis, 5; β_inv=1e-6)
    julia> predict_bayes(1.5, basis, μ, Σ_inv, 1e-6)
    (2.230543810972358, 2.3871769684628292)
    ```
"""
function predict_bayes(r::Float64, basis, μ, Σ_inv, β_inv)
    B = evaluate(basis, r)
    val = real(sum(μ .* B))
    σsq = sqrt(real(β_inv + B' * (Σ_inv \ B)))
    return val, σsq
 end
 

"""
    @overload
    predict_bayes(rr, args...)
    Returns the predicted value of bayesian ridge regression model with parameter μ, Σ_inv, precision β, at a prediction node r

    * @param rr             Vector{Float64}                 list of prediction nodes
    * @args...
        * @param basis      OrthPolyBasis1D3T{Float64}      basis functions
        * @param μ          Vector{ComplexF64}              [N-element] posterior distribution's average parameter of the brr model
        * @param Σ_inv      Matrix{ComplexF64}              [N × N] posterior distribution's average covariance matrix of the brr model
        * @param β_inv      Float64                         precision parameter of the brr model
    * @return               {Vector{Float64}, Vector{Float64}}      list of predicted value of brr model with bayesain uncertainty at the prediction nodes

    # Examples
    ```jldoctest
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> basis = chebyshev_basis(5)
    julia> μ, Σ_inv = brr(E, f1, [[1.0, 2.0]], basis, 5; β_inv=1e-6)
    julia> predict_bayes([1.5, 2.5], basis, μ, Σ_inv, 1e-6)
    ([2.230543810972358, 23.700662938903417], [2.3871769684628292, 16.322823621445494])
```
"""
 function predict_bayes(rr, args...)
     vals = Float64[]
     σs = Float64[]
     for r in rr
         v, σ = predict_bayes(r, args...)
         push!(vals, v)
         push!(σs, σ)
     end
     return vals, σs
 end

 """
    brr_error(NN, MM, K_R, testSampleSize, target_fn, averge_est_fn, basis, data_dst, dst; noise=0.0, test_uniform=false, prior="None", β_inv=1e-6, ρ=1, j=1, α=1.0)
    Returns a list of prediction error for the linear polynomail regression model given a list of degree of basis NN and a list of number of samples MM

    * @param NN                 Vector{Int64}   list of degree of basis
    * @param MM                 Vector{Int64}   list of number of samples
    * @param K_R                Int64           the number of radius distances per structure in the sample data
    * @param testSampleSize     Int64           the number of test data
    * @param target_fnc         generic fnc     target function
    * @param averge_est_fnc     generic fnc     average estimate function
    * @param basis_choice       String/function choice of basis
    * @param data_dst           Dict{String, Distributions.Censored{}}      a dict of data distribtuions
    * @param dst                String          chosen data distritbuion name for the train data
    * @keyargs noise            Float64         [default 0] noise in data
    * @keyargs test_uniform     Bool            [default false] whether the test data is genereated from unifrom distribution
    * @keyargs prior            String          [default "Identity" or "Hilbert", "Analy"] bayesain parameter smoothness prior assumption
    * @keyargs β_inv            Float64         [default 1e-6] precision parameter
    * @keyargs ρ                Int64           [default 1] analytical smoothness strength
    * @keyargs j                Int64           [default 1] algebraic smoothness strength
    * @keyargs α                Float64         [default 1.0] regularization coefficient
    * @return                   {Matrix{Float64}, Matrix{Float64}] [1 × length(MM), 1 × length(MM)] list of prediction error (RSME) and averaged uncertainty 
    
    # Examples
    ```jldoctest
    julia> MM=[40, 80, 160]; NN = [20 for _ in 1:length(MM)]; K_R = 2
    julia> data_dst = generate_data_dst(0.3, -1, 1, -1, 1)
    julia> dst = "uniform"
    julia> basis = get_basis("OrthPoly", NN[1], K_R, data_dst, dst)
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> brr_error(NN, MM, K_R, 100, f1, E, basis, data_dst, dst)
    ([6.907178501991979e-5 7.145647596802208e-5 6.071530206474897e-5], [0.002009668872050923 0.0013384897092909062 0.0011452814043087234])
    ```
"""
 function brr_error(NN, MM, K_R, testSampleSize, target_fn, averge_est_fn, basis, data_dst, dst; noise=0.0, test_uniform=false, prior="None", β_inv=1e-6, ρ=1, j=1, α=1.0)
    err = zeros(1, length(MM))
    var = zeros(1, length(MM))

    for i=1:length(NN)
        N = NN[i]
        M = MM[i]
        # generate data
        train, test = generate_data(M, K_R, testSampleSize, data_dst, dst)
        #get parameter
        μ, Σ_in = brr(averge_est_fn, target_fn, train, basis, N; noise=noise, prior=prior, β_inv=β_inv, ρ=ρ, j=j, α=α)

        if test_uniform
            test = sort(collect(Iterators.flatten([ rand(K_R)*2 .- 1 for _=1:testSampleSize ])))
        else
            test = sort(collect(Iterators.flatten(test)))
        end
        
        yp = target_fn.(test)
        y, σ = predict_bayes(test, basis, μ, Σ_in, β_inv)

        err[i] = norm(yp - y, 2)/sqrt(length(test))
        var[i] = norm(σ)/sqrt(length(test))
    end
    
    return err, var
end

"""
    brr_error_deg(NN, MM, K_R, testSampleSize, target_fn, averge_est_fn, basis_choice, data_dst, dst; noise=0.0, test_uniform=false, adaptedTrainSize=100, prior="None", β_inv=1e-6, ρ=1, j=1, α=1.0)
    Returns a list of prediction error for the linear polynomail regression model given a list of degree of basis NN and a list of number of samples MM

    * @param NN                 Vector{Int64}   list of degree of basis
    * @param MM                 Vector{Int64}   list of number of samples
    * @param K_R                Int64           the number of radius distances per structure in the sample data
    * @param testSampleSize     Int64           the number of test data
    * @param target_fnc         generic fnc     target function
    * @param averge_est_fnc     generic fnc     average estimate function
    * @param basis_choice       String/function choice of basis
    * @param data_dst           Dict{String, Distributions.Censored{}}      a dict of data distribtuions
    * @param dst                String          chosen data distritbuion name for the train data
    * @keyargs noise            Float64         [default 0] noise in data
    * @keyargs test_uniform     Bool            [default false] whether the test data is genereated from unifrom distribution
    * @keyargs adaptedTrainSize Int64           [default 100] number of train data to train the adapted basis
    * @keyargs prior            String          [default "Identity" or "Hilbert", "Analy"] bayesain parameter smoothness prior assumption
    * @keyargs β_inv            Float64         [default 1e-6] precision parameter
    * @keyargs ρ                Int64           [default 1] analytical smoothness strength
    * @keyargs j                Int64           [default 1] algebraic smoothness strength
    * @keyargs α                Float64         [default 1.0] regularization coefficient
    * @return                   {Matrix{Float64}, Matrix{Float64}] [1 × length(MM), 1 × length(MM)] list of prediction error (RSME) and averaged uncertainty 
    
    # Examples
    ```jldoctest
    julia> NN = [20, 40, 60]; MM= NN.^2; K_R = 2
    julia> data_dst = generate_data_dst(0.3, -1, 1, -1, 1)
    julia> dst = "uniform"
    julia> basis_choice = "OrthPoly"
    julia> f1(x) = abs(x)^3
    julia> E(f, Rs::Vector{Vector{Float64}}) = [ sum(f.(R))/length(R) for R in Rs ]
    julia> brr_error_deg(NN, MM, K_R, 100, f1, E, basis_choice, data_dst, dst)
    ([6.454644545323031e-5 4.497915515067082e-6 9.701265511072802e-7], [0.0010601873523998755 0.0010272452050097722 0.0010166471490359059])
    ```
"""
function brr_error_deg(NN, MM, K_R, testSampleSize, target_fn, averge_est_fn, basis_choice, data_dst, dst; noise=0.0, test_uniform=false, adaptedTrainSize=100, prior="None", β_inv=1e-6, ρ=1, j=1, α=1.0)
    Random.seed!(12216859)

    err = zeros(1, length(NN))
    var = zeros(1, length(NN))

    for i=1:length(NN)
        N = NN[i]
        M = MM[i]

        # generate data
        train, test = generate_data(M, K_R, testSampleSize, data_dst, dst)
        
        basis = get_basis(basis_choice, N, K_R, data_dst, dst; adaptedTrainSize)
        
        #get parameter
        μ, Σ_in = brr(averge_est_fn, target_fn, train, basis, N; noise=noise, prior=prior, β_inv=β_inv, ρ=ρ, j=j, α=α)

        if test_uniform
            test = sort(collect(Iterators.flatten([ rand(K_R)*2 .- 1 for _=1:testSampleSize ])))
        else
            test = sort(collect(Iterators.flatten(test)))
        end
        
        yp = target_fn.(test)
        y, σ = predict_bayes(test, basis, μ, Σ_in, β_inv)

        err[i] = norm(yp - y, 2)/sqrt(length(test))
        var[i] = norm(σ)/sqrt(length(test))
    end
    
    return err, var
end

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

function _design_matrix_pp(rs, basis, N)
    A = zeros(ComplexF64, length(rs), N)
    for (i, rr) in enumerate(rs)
      A[i, :] = sum(evaluate(basis, x(r)) * env(r) for r in rr) / length(rr)
    end 
    return A
end

function _design_matrix_pp_auto(rs, basis, N;  basic_model = false, prior="Identity", ρ=1, j=1, α=1.0)
    # for sklearn brr where "A = Γθ" -> Â = Γ⁻¹ A

    if basic_model
        Ψ = design_matrix(rs, basis, N)
    else
        Ψ = _design_matrix_pp(rs, basis, N)
    end
    
    # prior
    if prior == "Hilbert"
       Γ⁻¹ = prior_H(N, j, α)
    elseif prior == "Analy"
       Γ⁻¹ = prior_A(N, ρ, α)
    else
       Γ⁻¹ = prior_default(N, α)
    end
    
    # @show size(Ψ)
    # @show size(Γ⁻¹)

    return Ψ * Γ⁻¹
end

### Use ScikitLearn: https://github.com/ACEsuit/IPFitting.jl/blob/21617f5d6320d7aba5f87de98e56ef8af4dc2c72/src/lsq.jl#L545
function lsq_sk(averge_est_fn, target_fn, train, basis, N, noise; basic_model = false, solver=Dict("solver" => :qr, "prior" => "Identity"), verbose=true, ρ=1, j=1, α=1.0)
    Y = averge_est_fn(target_fn, train) .+ noise

    # Ψ = real(_design_matrix_pp(train, basis, N))
    Ψ = real(_design_matrix_pp_auto(train, basis, N; basic_model=basic_model, prior=solver["prior"], ρ=ρ, j=j, α=α))

    if solver["solver"] == :qr
        @info("Using QR")
        qrΨ = qr!(Ψ)
        κ = cond(qrΨ.R)

        verbose && @info("cond(R) = $κ")

        c = qrΨ \ Y

        rel_rms = norm(qrΨ.Q * (qrΨ.R * c) - Y) / norm(Y)
    
        verbose && @info("Relative RMSE on training set: $rel_rms")

        qrΨ = nothing
        Ψ = nothing
        GC.gc()

        return c

    elseif solver["solver"] == :brr
        BRR = pyimport("sklearn.linear_model")["BayesianRidge"]
        @assert haskey(solver, "brr_tol")
        if !haskey(solver, "brr_fit_intercept")
            fit_intercept=true
        else
            fit_intercept=solver["brr_fit_intercept"]
        end
        brr_n_iter = haskey(solver,"brr_n_iter") ? solver["brr_n_iter"] : 300
        brr_tol = solver["brr_tol"]

        verbose && @info("Using BRR: brr_n_iter=$(brr_n_iter), brr_tol=$(brr_tol)")

        clf = BRR(n_iter=brr_n_iter, tol=brr_tol, fit_intercept=fit_intercept, normalize=true, compute_score=true)
        clf.fit(Ψ, Y)

        if length(clf.scores_) < brr_n_iter
            verbose && @info "BRR converged to brr_tol=$(brr_tol) after $(length(clf.scores_)) iterations."
        else
            verbose && println()
            verbose && @warn "BRR did not converge to brr_tol=$(brr_tol) after brr_n_iter=$(brr_n_iter) iterations."
            verbose && println()
        end
        c = clf.coef_
        alpha = clf.alpha_
        lambda = clf.lambda_
        score = clf.scores_[end]
        verbose && @info("alpha=$(alpha), lambda=$(lambda), score=$(score)")

        rel_rms = norm(Ψ * c - Y) / norm(Y)

        return clf, c, alpha, lambda, score
    elseif solver["solver"] == :ard
        ARD = pyimport("sklearn.linear_model")["ARDRegression"]
        @assert haskey(solver, "ard_threshold_lambda")
        @assert haskey(solver, "ard_tol")
        if !haskey(solver, "ard_fit_intercept")
            fit_intercept=true
        else
            fit_intercept=solver["ard_fit_intercept"]
        end
        ard_n_iter = haskey(solver,"ard_n_iter") ? solver["ard_n_iter"] : 300
        ard_threshold_lambda = solver["ard_threshold_lambda"]
        ard_tol = solver["ard_tol"]
        verbose && @info("Using ARD: ard_n_iter=$(ard_n_iter), ard_tol=$(ard_tol), ard_threshold_lambda=$(ard_threshold_lambda)")

        clf = ARD(n_iter=ard_n_iter, threshold_lambda = ard_threshold_lambda, tol=ard_tol, fit_intercept=fit_intercept, normalize=true, compute_score=true)
        clf.fit(Ψ, Y)

        if verbose 
            if length(clf.scores_) < ard_n_iter
                @info "ARD converged to ard_tol=$(ard_tol) after $(length(clf.scores_)) iterations."
            else
                println()
                @warn "ARD did not converge to ard_tol=$(ard_tol) after ard_n_iter=$(ard_n_iter) iterations."
                println()
            end
        end

        c = clf.coef_
        alpha = clf.alpha_
        lambda = clf.lambda_
        score = clf.scores_[end]
        sigma = clf.sigma_
        
        # sigma_old = clf.sigma_

        # sigma = zeros(N, N)
        non_zero_ind = findall(x -> x != 0.0, c)
        # sigma[non_zero_ind[:,end], non_zero_ind] = sigma_old

        verbose && @info("Fit complete: keeping $(length(non_zero_ind)) basis functions ($(round(length(non_zero_ind)/length(c), digits=2)*100)%)")
        verbose && @info("score=$(score)")

        rel_rms = norm(Ψ * c - Y) / norm(Y)

        return clf, c, alpha, lambda, sigma, score
    end
end

end