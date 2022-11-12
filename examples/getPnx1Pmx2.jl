include("../src/utils.jl")

# form the average design matrix
max_degree = 20
num_sam = 100
K_Rs = 4
basis = legendre_basis

NN2b = [[i] for i = 1:max_degree]
poly = basis(max_degree, normalize = true)
X = rand(Uniform(-1, 1), (num_sam, K_Rs))
polyX1 = poly(X[:, 1])
@show size(polyX1)
polyX2 = poly(X[:, 2])
polyX3 = poly(X[:, 3])
polyX4 = poly(X[:, 4])

avg_design_matrix = zeros(num_sam, length(NN2b))
for i = 1:length(NN2b)
    avg_design_matrix[:, i] = sum([PX[:, i] for PX in [polyX1, polyX2, polyX3, polyX4]])
end

# calculate AtA
global AtA = avg_design_matrix' * avg_design_matrix

# eacm of AtA[n,m] = dot(avg[:,n],avg[:,m]). Therefore we remove sum(Pn(x1^(i))*Pm(x1^(i))+ Pn(x2^(i))*pm(x2^(i)) ... )
# from each AtA[n, m], where i runs through all the observations
for n = 1:max_degree
    for m = 1:max_degree
        global AtA
        thing = zeros(num_sam)
        for PX in [polyX1, polyX2, polyX3, polyX4]
            thing += PX[:, n] .* PX[:, m]
        end
        AtA[n, m] -= sum(thing)
    end
end


