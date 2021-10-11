### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e501c230-d8b5-11eb-37a3-e7c9ecf7bd80
begin
	using Plots
	using LinearAlgebra
	using Kronecker
	using PlutoUI
	using SparseArrays
	#https://juliapackages.com/p/kronecker
	
end

# ╔═╡ 95a5da6c-08b1-45ab-ac60-957f6671f1f3
TableOfContents()

# ╔═╡ eb3906c7-d1e5-479b-93c4-5b59707c55f1
md""" # Test Kronecker package"""

# ╔═╡ 736b4743-bcb3-42c7-8b4c-067f6a3f18b3
begin
	n = 2
	m = 3
	A = randn(n, n);
	B = rand(m, m);

	#v = rand(n*m);

	K = A ⊗ B
	
	v = rand(size(K,1),1)
	K * v  # equivalent with vec(B * reshape(v, 50, 100) * A')
#	trace(K)	
end

# ╔═╡ db75c98c-afe2-478b-9a93-9f87765e43cc
vec(K)

# ╔═╡ 27796834-1dfc-48b6-8005-c4dee17f3490
collect(vec(K))

# ╔═╡ 5c77981b-893a-4d77-b3db-3377afac0b63
begin

	function density_matrix(n, neg = 0, real_valued = false)
		#trace 1 remains preserved!
		dia = rand(n)
		if neg > 0
			dia[rand(1:n, neg)] .*= -1
		end
		dia = dia/sum(dia)
		if real_valued
			temp = rand(n,n)
		else
			temp = rand(n,n) + rand(n,n) * im
		end
		ew, ev = eigen(Hermitian(temp'*temp))
		d_mat = Hermitian(ev* Diagonal(dia) * ev')
		return d_mat
	end
	
	function unitary(n)
		A = (rand(n,n) .-0.5) + im*(rand(n,n) .-0.5)
		return eigvecs(A'+A)
	end
	
	
	function unity(i,n)
		E = spzeros(Bool,n,n)
		E[i] = true
    	return E
	end
	function partial_trace_matrix(dim_1, dim_2, traced_system)
# traces out the system indicated by traced_system
# for C = kron(A,B), tracing over the first system (traced_system = 1); 
# yields: reshape(p1*C[:], 4,4)/tr(A)  == B
# 
# in colex notation:
# traced_system == 1 represents a trace over index 2 and 4 and 
# traced_system == 2 represents a trace over index 1 and 3 and 
		if traced_system == 1
			T = [kron(sparse(I, dim_1,dim_1), unity(i,dim_2))[:] for i = 1:dim_2^2]
			col = reduce(vcat,[findnz(T[i])[1] for i = 1:dim_2^2])
			row = kron(1:dim_2^2, ones(Int64,dim_1))
			return sparse(row, col, trues(dim_2^2*dim_1))
		else
			T = [kron(unity(i, dim_1), sparse(I, dim_2, dim_2))[:] for i = 1:dim_1^2]
			col = reduce(vcat, [findnz(T[i])[1] for i = 1:dim_1^2])
			row = kron(1:dim_1^2, ones(Int64,dim_2))
			return sparse(row, col, trues(dim_1^2*dim_2))
			#return    Tr_B = [reshape(kron(reshape(unity(i,dim_1), dim_1, dim_1), I(dim_2)), 1, dim_2^2 * dim_1^2)[1,j] for i = 1:dim_1^2, j =1:dim_1^2*dim_2^2]
		end
	end


	function choi(A, dim_1)
		mm, nn = size(A)
		n = mm ÷ dim_1
		return reshape(permutedims(reshape(A,n,dim_1,n,dim_1),(1,3,2,4)), mm,mm)
	end

	function choi_rect(A,dim_1, dim_2, dim_3, dim_4)
		return reshape(permutedims(reshape(A, dim_1, dim_2, dim_3, dim_4), (1,3,2,4)), dim_1*dim_3, dim_2*dim_4)
	end
	
	
	function choi_index(i, n)
		return  (i.-1) .% n[1] .+ 1 .+
		n[1] .* ((i.-1) .% prod(n[1:3]) .÷ prod(n[1:2]) .+ 
		n[2] .* ((i.-1) .% prod(n[1:2]) .÷ n[1] .+
		n[3] .* ((i.-1) .÷ prod(n[1:3]))))
	end

	function ptr(ρ, dim_1, traced_system = 1)
		# get partial trace superoperator
		N,M = size(ρ)
		dim_2 = N÷dim_1
		PT = partial_trace_matrix(dim_1, dim_2, traced_system) 
		new_dim = traced_system == 1 ? dim_2 : dim_1
		# apply superoperator to vectorized density matrix and reshape to new dimensions
		σ = reshape(PT * ρ[:], new_dim, new_dim)
		return σ
	end

	function factorize(A, dim_1, dim_2)
		σ = ptr(A, dim_1, 2)
		ρ = ptr(A, dim_1, 1)
		A_tensor_prod = kron(σ, ρ)
		rho_c = A - A_tensor_prod
		return A_tensor_prod, rho_c
	end

	function normed_vector(k)
		v = (rand(k) .- 0.5) + im*(rand(k) .- 0.5)
		return v/norm(v)
	end

	function prob_dist(k)
		p = rand(k)
		return p/sum(p)
	end

md"""
# Functions
"""
end

# ╔═╡ ca92b1da-20cd-40b8-81ea-72c49f8898cd
	
function PPT(rho,dim_1)
		dim_2 = size(rho,1)÷dim_1
		if dim_1 *dim_2 > 6
			error("positive partial trace does not work for too large dimensions")
		end
		return eigvals(reshape(permutedims(reshape(rho,dim_1,dim_2,dim_1,dim_2), [1,4,3,2]),dim_1*dim_2, dim_1*dim_2))
		
end

# ╔═╡ 3a9e432b-7c7b-463b-bf13-e30c017b5880
function sep_state(k=1,n=2)
	
	VVV = sum([kron(normed_vector(n), normed_vector(n)) for _=1:k] .* (prob_dist(k)))
	return VVV / norm(VVV)
	
	
	
end

# ╔═╡ 627cb3ec-704b-4d9a-8a93-253b4c68971d
begin
	#test for k-block positivity
	# strategy: generate random matrices in 2x2 and test whether Choi matrix maps to positive operators, if true for all, then positive.
	# alternatively: check k-positivity: k as optional parameter. Create states with Schmidt rank k
	#Schmidt rank: 
	function pos_test(rho, k = 1, n = 2)
	
		runs = 10000
		test = zeros(runs)
		vector = zeros(ComplexF64, n^2)
		for i = 1:runs
			
			#sum([density_matrix(2) for _=1:k] * p_rand/sum(p_rand))

			VVN = sep_state(k,n)

			test[i] = real(VVN'* rho *VVN)

			if real(test[i]) < 0
				vector = VVN
			end
		end

		return sum(test .> 0) / runs
	
	end
	
end

# ╔═╡ 5450c94b-f78b-4bf2-b025-f035a28850cd
md"""
# Create a nice initial correlation term
"""

# ╔═╡ 2e11a379-2639-4cb9-a892-aa610680e671
md"
### Detecting entanglement with positive partial transpose
"

# ╔═╡ b54087af-d08c-402c-a4ff-52899b4787d1
begin
	# get r_c an arbitrary correlation term
	rr = density_matrix(4)
	r_t, r_c = factorize(rr,2,2)

	#partial trace of correlation term is zero
	
	#partial transpose is equal to swapping two indices in the four index picture
	
	
	r_t2, r_c2 = factorize(density_matrix(4),2,2)
	
	aaa, bbb = factorize(1/2*r_t + 1/2*r_t2,2,2)
	#convex combination of product states turn to be not product states
	if any(PPT(rr,2) .< 0)
		md"entanglement detected"
	else
		md"separable state"
	end
end

# ╔═╡ 29fbbc0a-3c97-4b15-9f20-3f66a562e2b7
md"
### get fully entangled state
"

# ╔═╡ cbb8399c-44c0-4def-8458-b6f34c94be66
begin
	dim_n = 2
	#note: check what for makes with the scope if combined with a markup element!
	let
		global ent = zeros(dim_n^2, dim_n^2)
		for k = 1:dim_n^2
			ent += kron(unity(k,dim_n), unity(k,dim_n))
		end
		ent = ent/tr(ent)
	end
	
	#apply a random weight:
	#sum(reshape(reduce(hcat, [kron(unity(k,dim_n), unity(k,dim_n)) for k = 1:4]),4,4,4) .* reshape([0.25, 0.3, 0.3, 0.2], 1,1,4), dims=3)
	
	md"get full entagled state"

end

# ╔═╡ 508b3c1e-1835-4575-a59e-bfc4f2504860
begin
	R = ent#0.5*kron(density_matrix(2), density_matrix(2)) + 0.5*kron(density_matrix(2), density_matrix(2))
	eigen(choi(R, 2))
	C = choi(R,2)
	
	
end


# ╔═╡ aca27f55-197e-4a27-98d5-be27981c3077
md"
### Create correlated initial state
"

# ╔═╡ 1cd9f410-b75a-4567-aeac-0809110be2d7
md" c23: $(@bind c23 Scrubbable(-1:0.1:1))"

# ╔═╡ 00647334-6d64-4cc3-95ab-f51196300893
md""" 
# Time evolution and dynamical map"""

# ╔═╡ cdde600f-d922-4376-81e8-949339378024
begin 
	diagonal_index = 1:(n+1):n^2
	# generate a unit basis of density operators
	rho_basis = zeros(ComplexF64,n^2,n^2)
	for k = 1:n
		for j = 1:n
			compl_factor = j>=k ? 1 : im
		temp = unity((k-1)*n + j, n)* compl_factor + unity(diagonal_index[k],n) + unity(diagonal_index[j],n)
		rho_basis[:,(k-1)*n+j] =((temp + temp')./(2*tr(temp)))[:]
		end
	end
	
end

# ╔═╡ 6724ccf7-dbd8-44cb-a96d-a7074b914a81
begin 
	#
md"""
## quantum--classical correlated initial state of the form

$\rho(t_0) = \sum_j p_j \Pi_j \otimes \rho_{B,j}$
	"""
	
end

# ╔═╡ 4a1ccfb2-bae6-464d-9790-0d09a4dcfbab
md""" $(@bind new_density Button("random density operator")) """

# ╔═╡ 7656ee8c-0b8c-402b-84b0-5517b39aac3a
begin
	new_density
	rho_a = density_matrix(2)# [0.5 0.1im; -0.1im 0.5]
	rho_b = density_matrix(2)
	#r1, r2 = factorize(density_matrix(4),2,2)
	
	#r2 does not produce any entanglement!!!
	r2 = [0 0.1 -0.1 -0.2; 0.1 0 -0.2 0.1; -0.1 -0.2 0 0.1; -0.2 0.1 0.1 0]
	#r2 = [0.25   0.0    0.0   0.5; 0.0   -0.25   0.0   0.0; 0.0    0.0   -0.25  0.0; 0.5    0.0    0.0   0.25]
	# for 0.7*r2 always non complete positive
	#rho = density_matrix(4) #kron(rho_a, rho_b) +  r2*0 # 
	#rho = 1/4*(kron(I(2), I(2)) + kron( τ₁ * σ₁ + τ₂ * σ₂ + τ₃ * σ₃, I(2)) - c23*kron(σ₂, σ₃))
	
	
	# Create quantum-classical correlated state
	Dim = 2
	
	RHO_b = [density_matrix(Dim) for _=1:Dim]
	
	RHO_s = density_matrix(Dim)
	evals, Pro = eigen(RHO_s)
	
	prob = prob_dist(Dim)
	
	rho = sum([kron(Pro[:,i] * Pro[:,i]', RHO_b[i]) for i = 1:Dim] .*prob)
	
end

# ╔═╡ 07199f9a-1180-4901-848e-3aca182cdf52
r2

# ╔═╡ 228654fc-8422-4f64-964c-c8b69c7fd175
eigvals(rho)

# ╔═╡ cc77f9ed-73ef-4129-9e6c-1a58dff2141d
if any(PPT(rho,2) .< 0)
		md"entanglement detected 🧬"
	else
		md"separable state 📎📎"
	end

# ╔═╡ 4e33f911-b190-4771-a1f4-2b15547fc84c
ptr(rho,Dim,2)

# ╔═╡ 56e4959e-586a-42b5-be90-8da1d719d6e5
RHO_b

# ╔═╡ 1289264a-dfaf-4889-a81e-2991295fde84
Pro[:,2]

# ╔═╡ 0c3439c3-948e-4cbd-965c-6b91bd10e11a
kron((Pro[1,:]*Pro[1,:]'),I(3)) 

# ╔═╡ 36c5e484-275b-4325-8a45-89e23d9f48f8
Pro[:,1] * Pro[:,1]'

# ╔═╡ 519bfd98-a91b-496b-a484-8fa861aef467
eigvals(rho)

# ╔═╡ a421f989-e88d-45ab-a9b8-e79fa321e2d7
if any(PPT(rho,Dim) .< 0)
		md"entanglement detected 🧬"
	else
		md"separable state 📎📎"
	end

# ╔═╡ b57f0f09-831b-4c99-ba6a-7b49af111c9b
tr(sqrt(choi(rho,Dim)' * choi(rho,Dim))) # cross norm criteria or realignment criteria (with trace norm)

# ╔═╡ c1906f46-ffd6-4c26-909b-d952066813d9
md"""
``\boldsymbol{K}_i = \sqrt{\varepsilon_{k'}} \langle b_k | U(t,t_0) | b_{k'} \rangle ``
"""

# ╔═╡ bc6dea59-2db2-4d9a-aca9-3ae77159a070
md"""
Time: $(@bind time Slider(0:0.1:40, show_value=true, default = 0.5))
	
	"""

# ╔═╡ d8b26c46-2c3b-4e8c-bfad-f81a0406ac6e
time

# ╔═╡ ec05b0c0-92d4-44ef-80be-ef3d3e53e688
time_vector = 0:0.1:50

# ╔═╡ deed3f43-ddbf-47fc-abb7-3fdfe8016ba9
begin
	md""" 
	Energy of first system site $(@bind eps_1 Slider(-4:0.01:4, show_value=true, default = 1))
	
	Second system: $(@bind eps_2 Slider(-4:0.01:4, show_value=true, default = 2))
	
	Hopping: $(@bind t Slider(-4:0.01:4, show_value=true, default = 0.5))
	
	"""
end

# ╔═╡ 6a3574a9-9797-49b3-891f-d04b4919614f
begin
	eps_3 = 3
	eps = [0, eps_1, eps_2, eps_3]

	if Dim == 2

	H_1 = [0 0 0 0; 0 eps[2] 0 0; 0 0 eps[3] 0; 0 0 0 eps[2] + eps[3]]
	H_2 = [0 0 0 0; 0 0 t 0; 0 t 0 0; 0 0 0 0]
	H_full= H_1 + H_2
	
	U = exp(-im * H_full * time)
	else
		U = unitary(Dim^2)
	end
	md" time slider: $(time)"
end

# ╔═╡ 90a2efc4-1706-4882-a1d4-c41f42f9c230
begin 
	bath = 2 # means colex 2 and 4
	# bath == 2 -> choi is defined differently (add transpose)
	# choose rather colexicographical order also for tensor product!
	# A ⊗ B -> kron(B,A)
	# TrB -> ptr(. ,1)
	# factorize??? -> swap arguments in kron?? should work perfectly as it is!
	# 
	system = 1 # means coles 1 and 3
	
	#decompose initial:
	sigma_rho_0, rho_c_0 = factorize(rho,Dim,bath)
	# time evolution
	
	aim_density = ptr(U * rho * U',Dim,bath)
	
	sigma_c = ptr(U * rho_c_0 * U',Dim,bath)
	
	#not necessary anymore
	#=
	Lambda = repeat(sigma_c[:],1,n^2) * inv(rho_basis)
	u_svd, z_svd, v_svd = svd(Lambda)
	#lv_inv = lv^-1
	temp_vec = u_svd[:,real(z_svd) .== maximum(real.(z_svd))]
	temp_vec_2 = v_svd[:,real(z_svd) .== maximum(real.(z_svd))]
	Proj = temp_vec * temp_vec_2' * maximum(z_svd)
	=#
	Choi_c = kron(I(Dim), sigma_c)#choi(Proj,2)
	
	
	# compare UDM
	sigma = ptr(U * sigma_rho_0 * U',Dim,bath)
	sigma_full = ptr(U * rho * U',Dim,bath)
	# create Krausoperators
	rho_b_0 = ptr(sigma_rho_0,Dim,system)
	sigma_0 = ptr(sigma_rho_0,Dim,bath)
	
	
	bw, bv = eigen(rho_b_0)	
	Kr = [zeros(ComplexF64,Dim,Dim) for _=1:Dim^2] 
	Up = [zeros(ComplexF64,Dim,Dim) for _=1:Dim,_=1:Dim] 
	for k = 1:Dim
		for j = 1:Dim
			if bath == 1
				Up[k,j] = U[(1:Dim).+(k-1)*Dim, (1:Dim).+(j-1)*Dim] #U[k:2:end,j:2:end] #
			else
				Up[k,j] = U[k:Dim:end,j:Dim:end]
			end
		end
	end
	
	for k = 1:Dim, j=1:Dim, g=1:Dim, h=1:Dim
		  Kr[(k-1)*Dim + j] += bv[g,j]' * Up[g,h] * bv[h,k] * sqrt(bw[k])
		
	end
	#create Choi of UDM
	
	Choi_t = sum([Kr[k][:] * Kr[k][:]' for k = 1:Dim^2])
	#alternative
	if bath == 2
		Choi_t_2 = transpose(choi(U,Dim)) * kron(rho_b_0, I(Dim)) * transpose(choi(U,Dim))'
	else
		Choi_t_2 = choi(U,Dim) * kron(rho_b_0, I(Dim)) * choi(U,Dim)'
	end
	
	sigma_dyn_t = reshape(choi(Choi_t + Choi_c,Dim) * sigma_0[:],Dim,Dim)
	
end

# ╔═╡ 09519cbd-1821-4453-ac28-db81f0adc701
norm(sigma_full - sigma_dyn_t) #choi superoperator from inhomogeneous and homogeneous part vs full time evolution

# ╔═╡ 1f7e4796-c90f-4801-b98c-12518199e1c3
norm(Choi_t_2 - Choi_t)

# ╔═╡ 586e00d9-6144-4fec-b0e6-77465483a78a
eigvals(Choi_t + Choi_c)

# ╔═╡ 256c69ba-d43e-4571-8269-54885157d67d
round.(rho_c_0,digits=5)

# ╔═╡ 5c15961d-339a-4511-a441-f11cb9bc7399
begin
	choi_evals = real(eigvals(Choi_t + Choi_c))
if any(choi_evals .< 0)
	#eigvals(sum([Kr[k][:] * Kr[k][:]' for k = 1:4]) + Choi_c)
	md"not completely positive 😈 $(string(round.(choi_evals,digits=4))) "
else
	md" completely positivie 😇 $(string(round.(choi_evals,digits=4)))"
end
end

# ╔═╡ 41d63a63-5dff-4b99-b343-a23124922e07
round.(choi_evals,digits=5)

# ╔═╡ e5b9cb77-da33-45e5-827e-49f768cd1f07
pos_test(Choi_t + Choi_c,1,Dim)

# ╔═╡ be0fb311-f271-40ca-856d-184d98c8c0e7
real(eigvals(reshape(permutedims(reshape(Choi_t + Choi_c,Dim,Dim,Dim,Dim), [3,2,1,4]), Dim^2,Dim^2)))

# ╔═╡ aa527940-4029-48ce-abed-fc2695bd3900
real.(Choi_t)

# ╔═╡ 80a80124-14c2-4cdb-bbd3-16951a06e2f4
sigma_dyn_t

# ╔═╡ 0b4788d7-0c22-4821-b355-19b31aadfdf1
sigma_full

# ╔═╡ 8bcc43e5-00dd-400e-ad15-db0358c9dce2
sigma

# ╔═╡ 331505b8-6195-4ba3-a8cc-abe217c98867
sigma_0

# ╔═╡ 71efd276-b9ed-4d9c-a902-1e028ace8114
begin
	rho_try = zeros(ComplexF64,2,2)
	for k = 1:4
		rho_try += Kr[k] * sigma_0 * Kr[k]'
	end
	test = zeros(ComplexF64,2,2)
	for k = 1:4
		test += Kr[k]' * Kr[k]
	end
	test
end

# ╔═╡ cc741913-140c-4efb-ab3a-27520fb35563
rho_try

# ╔═╡ 2fe62982-40af-4684-a519-1a520892a41c
aim_density

# ╔═╡ 574eda40-33f0-4a03-8d23-103db03614d3
begin 
	Pro
	# try to find the composition of the Choi operator for the quantum--classical correlated initial state
	RHO_b
	
	
	KK = [[zeros(ComplexF64,Dim,Dim) for _=1:Dim^2] for _=1:length(RHO_b)]
	
	for h = 1:length(RHO_b)
	
		local bw, bv = eigen(RHO_b[h])	
		local Kr = [zeros(ComplexF64,Dim,Dim) for _=1:Dim^2] 
		local Up = [zeros(ComplexF64,Dim,Dim) for _=1:Dim,_=1:Dim] 
		for k = 1:Dim
			for j = 1:Dim
				if bath == 1
					Up[k,j] = U[(1:Dim).+(k-1)*Dim, (1:Dim).+(j-1)*Dim] #U[k:2:end,j:2:end] #
				else
					Up[k,j] = U[k:Dim:end,j:Dim:end]
				end
			end
		end

		for k = 1:Dim, j=1:Dim, g=1:Dim, h=1:Dim
			  Kr[(k-1)*Dim + j] += bv[g,j]' * Up[g,h] * bv[h,k] * sqrt(bw[k])

		end
	KK[h] = Kr
	end
	
	test_1 = sum([sum([(KK[h][i]*Pro[:,h]*Pro[:,h]')[:] * (KK[g][i]*Pro[:,g]*Pro[:,g]')[:]' for h=1:3, g=1:3]) for i=1:9])
	KKK = [sum([KK[h][i] * (Pro[:,h]*Pro[:,h]') for h=1:3]) for i=1:9]
	
	test_2 = sum([KKK[i][:] * KKK[i][:]' for i = 1:9])
end

# ╔═╡ 40777c5b-f031-404a-8c68-4dc7199245eb
norm(reshape(choi(test_2,Dim) * sigma_0[:],Dim,Dim) - sigma_dyn_t)

# ╔═╡ 87abb02f-d6cf-469d-af1f-e0433b19e572
test_2

# ╔═╡ 0a661954-8cd4-4dbd-a2b6-d3a759dc15a1
test_1

# ╔═╡ 9cca546c-2773-47f5-a45e-831ffa0d886c
sum([ kron( Pro[:,i] * Pro[:,i]', I(Dim)) * choi(U,Dim) * kron(RHO_b[i], RHO_b[j]) * choi(U,Dim)' * kron(I(Dim), Pro[:,j] * Pro[:,j]' )  for i = 1:Dim, j=1:Dim])

# ╔═╡ ab40cfbc-51e4-4d5a-bf97-db4ef3e5c002
begin 
	
	# relation of partial trace for initial product states - works! :D
	abs.((choi((U),2) * kron(rho_b_0,I(2)) * choi(U,2)') - (Choi_t))
	

end

# ╔═╡ b105e3f9-7387-4999-a3af-cbdda12eb9ed
rho_t = [exp(-im * H_full * i) * sigma_rho_0 * exp(im * H_full * i) for i = time_vector]

# ╔═╡ d8c75acc-e6c7-443d-9ad8-e804aec54349
rho_r = reshape(reduce(hcat, rho_t),4,4,length(time_vector));

# ╔═╡ f449c516-435e-428c-afad-1e4315cf7e99
time_slider = @bind tt Slider(1:length(time_vector), show_value = true, default = 5)

# ╔═╡ fc9b2490-233d-44b9-8ddf-6368c4f74bae
time_slider

# ╔═╡ a141a91d-5e12-4633-8318-b7f91dc19c46
begin
	sys_1 = reshape(reduce(hcat,[ptr(rho_r[:,:,i],2,2) for i = 1:length(time_vector)]),2,2,length(time_vector))
	
	
	plot(time_vector, real(sys_1[1,1,:]))
	plot!(time_vector, real(sys_1[2,2,:]))
	plot!(time_vector, real(sys_1[1,2,:]))
	plot!(time_vector, imag(sys_1[1,2,:]))

	
end

# ╔═╡ c94aefb0-42e0-46eb-9274-055378adeae2
sys_1[:,:,1]

# ╔═╡ 6ed3ba14-19bf-4e35-933e-5740bbaa227b
sys_1[:,:,tt]

# ╔═╡ 46e923aa-60a2-4458-b26f-47edaa4b53e0
begin
	factors = [factorize(rho_r[:,:,i],2,2) for i = 1:length(time_vector)]
	c = [sum(abs.(factors[i][2]).^2) for i = 1:length(time_vector)]
	
end

# ╔═╡ 6ed477ad-a5f8-4682-b171-323c096c9748
begin
	
	plot(time_vector, real.(rho_r[2,2,:]))
	plot!(time_vector, real.(rho_r[3,3,:]))
	plot!(time_vector, real([tr(ptr(rho_r[:,:,i],2,2)^2) for i = 1:length(time_vector)]), ylim = [0,0.7], label = "purity of sys 1")
		plot!(time_vector, c)
	
end

# ╔═╡ 484a0ec1-a352-473e-9826-e7d38d1de570
sys_1[:]

# ╔═╡ 970f944e-e57a-43cd-9e3e-a129d4ffb53d
size(time_vector)

# ╔═╡ 15493335-e1b7-489e-95c4-5f74a2e23534
size(rho_r[3,3,:])

# ╔═╡ 6df2ffbf-f921-477f-b6bf-b8d2070f9cc2
time_vector

# ╔═╡ 7b5f91e5-f5c4-4971-b62f-7ecf5c5393c6
md" # Partial trace"

# ╔═╡ 733ad294-2c04-4bbb-bf5f-778a5bed34ca
begin
	dim_1 = 3
	a = rand(dim_1,dim_1)
	b = rand(2,2)
	G = kron(a,b)
end

# ╔═╡ b4f086ea-5f62-42e9-a7d0-02dff480fdcf
begin
	function  partial_trace(K, dim_1, which_space)
	dim_2 = size(K,1) ÷ dim_1
	# check QuantumOptics.jl
	#https://qojulia.org
	if which_space == 1
		L = kron(I(dim_1), trues(dim_2, dim_2))
		P = dropdims(sum(reshape(K[L],dim_2, dim_2, dim_1),dims=3),dims=3)
		
	else
		L = kron(trues(dim_1,dim_1), I(dim_2))
		P = dropdims(sum(reshape(K[L], dim_1, dim_2, dim_1),dims=2),dims=2)
		
	end
		
	return P
	end
	
	
	

end

# ╔═╡ b50bde64-a115-4abe-a952-88983c9fa511
@time partial_trace(G,2,2)


# ╔═╡ 992f454a-4e30-4080-86c0-82aeec7d9597
partial_trace(G,3,1)/tr(a) - b

# ╔═╡ a2d4e774-6c12-4c2c-a213-44cedc6c1667
begin
md"""
# Issue of how to sum over a 3rd dimension
There are three versions:
1. dropdims(sum(A,dims=3),dims=3)
2. sum(eachslice(A,dims=3))
3. Using a function (here called f)
"""	
end

# ╔═╡ 7ce9332f-3639-465f-bea1-417308914c47
function f(x::AbstractArray{<:Number, 3})
           nrows, ncols, n = size(x)
           result = zeros(eltype(x), nrows, ncols)
           for k in 1:n
               for j in 1:ncols
                   for i in 1:nrows
                       result[i,j] += x[i,j,k]
                   end
               end
           end
           result
       end

# ╔═╡ b77e9684-fb8a-4d71-9820-f11654508850
begin
	x = rand(3,3,100000)
	@time dropdims(sum(x, dims=3), dims=3)
	@time sum(eachslice(x,dims=3))
	@time f(x)
	
	md""" Check result in the console"""
end

# ╔═╡ 4ed1a61c-a916-46f2-abb2-f95837db3149
dim_2 = size(K,2)÷dim_1


# ╔═╡ bf12a345-7f36-41cf-8567-84819a99e17a
md"""
# Parametrizing a qubit
"""

# ╔═╡ 7fa6e288-20bd-468f-8fb4-bb50cb4ee548
p = 1

# ╔═╡ 8a6325b6-8c55-4418-8163-72210a9ef3b4
md"""
# Tensor product of superoperators 
``\mathcal{T} \otimes \mathcal{S}``
"""

# ╔═╡ 4ae7f11e-728c-47b6-a810-39ce5bccd1ea
begin
	Tsup = rand(4,4) + 1im*rand(4,4)
	
	Ssup = rand(4,4) + 1im * rand(4,4)
	
	X_T = [rand(2,2) for _=1:5]
	X_S = [rand(2,2) for _=1:5]
	
	X = sum([kron(X_S[i], X_T[i]) for i = 1:5])
	
	
	#elementwise
	final_sol = sum([kron(reshape(Ssup * X_S[i][:],2,2),reshape(Tsup * X_T[i][:],2,2)) for i = 1:5])
	# final version with 
	CC = sparse(1:16, choi_index(1:16,[2,2,2,2]),ones(16))
	
	# full operator
	T_total = kron(Ssup, Tsup)
	T_total = T_total[choi_index(1:16,[2,2,2,2]),choi_index(1:16,[2,2,2,2])]
	final_vergleich = reshape(T_total*X[:], 4,4)
	final_2 = reshape(CC*kron(Ssup, Tsup) * CC * X[:],4,4)
end

# ╔═╡ 7b80e86a-3009-42d3-9ea9-8e9e245fecd5
final_sol

# ╔═╡ 103372c5-49aa-44d7-ae79-5af08892c5b1
CC * kron(I(4), Tsup) * CC

# ╔═╡ 770d537d-dc21-45fa-8ab5-55554201e479
Tsup

# ╔═╡ 146252a2-834d-4ead-9358-f3e520dc54a1
reshape( kron(I(4), Tsup) *  I(4)[:],4,4)

# ╔═╡ ff6fdcad-1196-468a-bfe3-52e406c5decb
norm(final_sol - final_vergleich)

# ╔═╡ a4fd9c40-4e99-4f5c-b9b4-6078611407dc
md""" # Examine the tensor decomposition of a Hamiltonian"""

# ╔═╡ 89bce87e-b4db-4726-bb9e-e67dab205089
begin 
	H = rand(6,6) .+5
	H = H+H'
end

# ╔═╡ 403d03b1-e6d4-423a-94b4-05dd6fa37ebc
tr(H)

# ╔═╡ 733acbdd-c57f-4e05-a522-be97f7ef2ae6
@bind trb Slider(-10:0.1:20, show_value = true)

# ╔═╡ 6834d85e-271f-4994-932c-919ca5cd36f8
tr(H)/4

# ╔═╡ e18bdab5-458d-4aaa-bbf4-06b07d2071f7
	begin
	d_1 = 3
	d_2 = size(K,1) ÷ d_1
end

# ╔═╡ dff95946-ce40-469b-83a4-0c3459840d98
#@bind trs Slider(-10:0.1:10, show_value = true)
trs = 1/d_2 * (tr(H) - d_1 * trb)

# ╔═╡ 30d18139-5c12-4ac1-b196-63bcd90ef5d7
H_S = (partial_trace(H,d_1,2) - I(d_1)*trb)./d_2

# ╔═╡ cda2c9cd-935e-4ca0-bd4a-a99caf68b5a7
tr(H_S) - trs

# ╔═╡ da36ec94-fa8a-4927-8b37-756c4a0bdba8
H_B = (partial_trace(H,d_1,1) - I(d_2) * trs)./d_1

# ╔═╡ a9f16b3c-bbc5-4f0d-84cf-a9ee59c986b4
tr(H_B) - trb

# ╔═╡ bd40b6dd-3441-4f76-9983-73b6fea43a62
H_I = H- kron(H_S, I(d_2)) - kron(I(d_1), H_B)

# ╔═╡ 250fcce2-7a9d-417a-8aaa-f407450ffb09
tr(H_I)

# ╔═╡ 7d157865-23f7-46a9-b04f-7e61ce3085e5
rank(H_I)

# ╔═╡ c31f3bc1-f78d-49e7-99a3-e21fcf2abb2a
H_I * H - H * H_I

# ╔═╡ c330dc0f-b941-4ece-ba1c-0a8301aa7d33
partial_trace(H_I, d_1,1)

# ╔═╡ e48108d1-1507-4ed5-bb35-6d5b18f83c76
begin 

	 trb_ = -41:0.1:40
	trs_ = -40:0.11:40
	z = zeros(length(trb_), length(trs_))
	for i = 1:length(trb_)
		for j = 1:length(trs_)
		trs = trs_[j]
		trb = trb_[i]
		H_S = (partial_trace(H,d_1,2) - I(d_1)*trb)./d_2
		H_B = (partial_trace(H,d_1,1) - I(d_2) * trs)./d_1
		z[i,j] = abs(tr(H_B) - trb) + abs(tr(H_S) - trs)
		end
	end
	
end

# ╔═╡ 901ac58c-24e1-4242-9a80-c1925cb36184
partial_trace(H,dim_1,1)

# ╔═╡ ec72a849-17df-40c7-aafc-390083fdfd6e
begin 
	contour(trs_, trb_, z)
	plot!(trs_, trs_ .* -d_2 ./d_1 .+ tr(H)/d_1)
end

# ╔═╡ 62fbffea-699c-44e8-a1fe-6b957c641a61


# ╔═╡ 75ff52ac-8832-4a47-aa56-81d9083348bb
minimum(z[:])

# ╔═╡ 19d8a1f4-8bf2-4f18-a8cd-1f1e4d12d143
tr(H)

# ╔═╡ fa193035-ff5c-4334-b65f-695445db4160
partial_trace(H,2,1)

# ╔═╡ 676bb8c8-5e13-4c49-bed4-62fd92e44779
partial_trace(H,2,2)

# ╔═╡ 7b1ad86e-1ca0-4666-8807-32f60b1e6191
begin
scrub_tau_1 = @bind τ₁ Scrubbable(-1:0.1:1)
scrub_tau_2 = @bind τ₂ Scrubbable(-1:0.1:1)
scrub_tau_3 = @bind τ₃ Scrubbable(-1:0.1:1)
	
end

# ╔═╡ 9ce64a5b-05fe-4bed-b2c4-044053255a77
begin
md"""
``\tau_1``: $(scrub_tau_1)
``\tau_2``: $(scrub_tau_2)
``\tau_3``: $(scrub_tau_3)
	"""
end

# ╔═╡ 359dc820-3ee9-4633-99e4-4e4eb76b6548
begin
md"""
``\tau_1``: $(scrub_tau_1)
``\tau_2``: $(scrub_tau_2)
``\tau_3``: $(scrub_tau_3)
	
To gain a valid density matrix the parameters ``τᵢ`` have to lie within a sqhere of radius 0.5.
Those parameter sets on that sphere yield pure states.
"""
	
end

# ╔═╡ 6b666e20-e9f2-4c3c-adb4-9a556139dc20
begin
σ₁ = [0 1; 1 0]
σ₂ = [0 -im; im 0]
σ₃ = [1 0; 0 -1]
qubit = Diagonal([0.5, 0.5]) + τ₁ * σ₁ + τ₂ * σ₂ + τ₃ * σ₃

	
end

# ╔═╡ 69ef6ce0-cd38-41a8-9463-1326c42d29e4
begin
	pauli = collect([I(2)[:] σ₁[:] σ₂[:] σ₃[:]])
	choi(repeat(σ₃[:], 1,4) * rho_basis^-1,2)
	
	
end

# ╔═╡ f2e6d34f-13c4-4e6e-b1c7-09d63d660c36
kron(I(2),σ₂)

# ╔═╡ 92755d15-ea82-4313-8e67-945d777080c0
σ₁

# ╔═╡ a09f4eea-50ff-4307-96a9-811ec1148949
begin
	σₚ = (I(2) + σ₃)/2
	σₘ = (I(2) - σ₃)/2
	#note, that D is the transposition operator (non CP)
	D = -p*(im* σ₂[:]) * (im * σ₂[:])'/2 +(1+p)/2*( σₚ[:] * σₚ[:]' + σₘ[:] * σₘ[:]' + σ₁[:] * σ₁[:]'/2 )
	eigvals(D)
	
	#p3 = sqrt(0.5^2 - τ₁^2 - τ₂^2 )
	D2 = kron(I(2), τ₁ * σ₁ + τ₂ * σ₂ + τ₃ * σ₃)
	# get affine part!!
end

# ╔═╡ a9cb62af-091f-472f-892e-c0aca842e5b1
begin
	#check whether \sum K' * K = 1
	(1+p)/2 * (σ₁' * σ₁/2 + σₘ' * σₘ + σₚ' * σₚ) - p*(im * σ₂)' * (im* σ₂)/2
end

# ╔═╡ 2b41dc34-d3d9-4736-8511-17049f87e604
sum(D, dims = 1)

# ╔═╡ 7dff4447-10fa-46a5-a1ac-0a64962449c3
tr(reshape(choi(D2,2) * density_matrix(2)[:],2,2))

# ╔═╡ e080aaf6-1453-4142-91cc-7ee26dd68f13
sqrt(τ₁^2 + τ₂^2 + τ₃^2)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Kronecker = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Kronecker = "~0.4.3"
Plots = "~1.18.2"
PlutoUI = "~0.7.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random", "StaticArrays"]
git-tree-sha1 = "ed268efe58512df8c7e224d2e170afd76dd6a417"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.13.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b83e3125048a9c3158cbb7ca423790c7b1b57bea"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.57.5"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e14907859a1d3aee73a019e7b3c98e9e7b8b5b3e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.57.3+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "15ff9a14b9e1218958d3530cc288cf31465d9ae2"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.3.13"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "47ce50b742921377301e15005c96e979574e130b"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.1+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "c6a1fff2fd4b1da29d3dccaffb1e1001244d844e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.12"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[Kronecker]]
deps = ["FillArrays", "LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "90e082a267982069e624ea0f825d324c86a01b4e"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.4.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedDims]]
deps = ["AbstractFFTs", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "52985b34519b12fd0dcebbe34e74b2dbe6d03183"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.35"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "f32cd6fcd2909c2d1cdd47ce55e1394b04a66fe2"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.18.2"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "a43a7b58a6e7dc933b2fa2e0ca653ccf8bb8fd0e"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.6"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2f6792d523d7448bbe2fec99eca9218f06cc746d"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.8"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "8ed4a3ea724dac32670b062be3ef1c1de6773ae8"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.4"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═e501c230-d8b5-11eb-37a3-e7c9ecf7bd80
# ╠═95a5da6c-08b1-45ab-ac60-957f6671f1f3
# ╟─eb3906c7-d1e5-479b-93c4-5b59707c55f1
# ╠═736b4743-bcb3-42c7-8b4c-067f6a3f18b3
# ╠═fc9b2490-233d-44b9-8ddf-6368c4f74bae
# ╠═db75c98c-afe2-478b-9a93-9f87765e43cc
# ╠═27796834-1dfc-48b6-8005-c4dee17f3490
# ╠═5c77981b-893a-4d77-b3db-3377afac0b63
# ╠═ca92b1da-20cd-40b8-81ea-72c49f8898cd
# ╠═3a9e432b-7c7b-463b-bf13-e30c017b5880
# ╠═627cb3ec-704b-4d9a-8a93-253b4c68971d
# ╟─5450c94b-f78b-4bf2-b025-f035a28850cd
# ╟─2e11a379-2639-4cb9-a892-aa610680e671
# ╠═b54087af-d08c-402c-a4ff-52899b4787d1
# ╟─29fbbc0a-3c97-4b15-9f20-3f66a562e2b7
# ╠═cbb8399c-44c0-4def-8458-b6f34c94be66
# ╠═508b3c1e-1835-4575-a59e-bfc4f2504860
# ╟─aca27f55-197e-4a27-98d5-be27981c3077
# ╠═7656ee8c-0b8c-402b-84b0-5517b39aac3a
# ╟─1cd9f410-b75a-4567-aeac-0809110be2d7
# ╟─9ce64a5b-05fe-4bed-b2c4-044053255a77
# ╠═07199f9a-1180-4901-848e-3aca182cdf52
# ╠═228654fc-8422-4f64-964c-c8b69c7fd175
# ╠═cc77f9ed-73ef-4129-9e6c-1a58dff2141d
# ╟─00647334-6d64-4cc3-95ab-f51196300893
# ╠═4e33f911-b190-4771-a1f4-2b15547fc84c
# ╠═6a3574a9-9797-49b3-891f-d04b4919614f
# ╠═cdde600f-d922-4376-81e8-949339378024
# ╠═69ef6ce0-cd38-41a8-9463-1326c42d29e4
# ╠═f2e6d34f-13c4-4e6e-b1c7-09d63d660c36
# ╠═d8b26c46-2c3b-4e8c-bfad-f81a0406ac6e
# ╠═90a2efc4-1706-4882-a1d4-c41f42f9c230
# ╠═09519cbd-1821-4453-ac28-db81f0adc701
# ╠═1f7e4796-c90f-4801-b98c-12518199e1c3
# ╟─6724ccf7-dbd8-44cb-a96d-a7074b914a81
# ╠═586e00d9-6144-4fec-b0e6-77465483a78a
# ╠═40777c5b-f031-404a-8c68-4dc7199245eb
# ╠═574eda40-33f0-4a03-8d23-103db03614d3
# ╠═56e4959e-586a-42b5-be90-8da1d719d6e5
# ╠═87abb02f-d6cf-469d-af1f-e0433b19e572
# ╠═9cca546c-2773-47f5-a45e-831ffa0d886c
# ╠═0a661954-8cd4-4dbd-a2b6-d3a759dc15a1
# ╠═1289264a-dfaf-4889-a81e-2991295fde84
# ╠═0c3439c3-948e-4cbd-965c-6b91bd10e11a
# ╠═36c5e484-275b-4325-8a45-89e23d9f48f8
# ╠═256c69ba-d43e-4571-8269-54885157d67d
# ╠═5c15961d-339a-4511-a441-f11cb9bc7399
# ╟─4a1ccfb2-bae6-464d-9790-0d09a4dcfbab
# ╠═e5b9cb77-da33-45e5-827e-49f768cd1f07
# ╟─519bfd98-a91b-496b-a484-8fa861aef467
# ╠═be0fb311-f271-40ca-856d-184d98c8c0e7
# ╟─a421f989-e88d-45ab-a9b8-e79fa321e2d7
# ╠═b57f0f09-831b-4c99-ba6a-7b49af111c9b
# ╠═41d63a63-5dff-4b99-b343-a23124922e07
# ╠═c1906f46-ffd6-4c26-909b-d952066813d9
# ╠═bc6dea59-2db2-4d9a-aca9-3ae77159a070
# ╠═ab40cfbc-51e4-4d5a-bf97-db4ef3e5c002
# ╠═aa527940-4029-48ce-abed-fc2695bd3900
# ╠═80a80124-14c2-4cdb-bbd3-16951a06e2f4
# ╠═0b4788d7-0c22-4821-b355-19b31aadfdf1
# ╠═8bcc43e5-00dd-400e-ad15-db0358c9dce2
# ╠═cc741913-140c-4efb-ab3a-27520fb35563
# ╠═331505b8-6195-4ba3-a8cc-abe217c98867
# ╠═71efd276-b9ed-4d9c-a902-1e028ace8114
# ╠═2fe62982-40af-4684-a519-1a520892a41c
# ╠═ec05b0c0-92d4-44ef-80be-ef3d3e53e688
# ╠═b105e3f9-7387-4999-a3af-cbdda12eb9ed
# ╠═d8c75acc-e6c7-443d-9ad8-e804aec54349
# ╟─deed3f43-ddbf-47fc-abb7-3fdfe8016ba9
# ╠═c94aefb0-42e0-46eb-9274-055378adeae2
# ╠═6ed3ba14-19bf-4e35-933e-5740bbaa227b
# ╠═f449c516-435e-428c-afad-1e4315cf7e99
# ╠═a141a91d-5e12-4633-8318-b7f91dc19c46
# ╠═46e923aa-60a2-4458-b26f-47edaa4b53e0
# ╠═6ed477ad-a5f8-4682-b171-323c096c9748
# ╠═484a0ec1-a352-473e-9826-e7d38d1de570
# ╠═970f944e-e57a-43cd-9e3e-a129d4ffb53d
# ╠═15493335-e1b7-489e-95c4-5f74a2e23534
# ╠═6df2ffbf-f921-477f-b6bf-b8d2070f9cc2
# ╟─7b5f91e5-f5c4-4971-b62f-7ecf5c5393c6
# ╠═733ad294-2c04-4bbb-bf5f-778a5bed34ca
# ╠═b4f086ea-5f62-42e9-a7d0-02dff480fdcf
# ╠═b50bde64-a115-4abe-a952-88983c9fa511
# ╠═992f454a-4e30-4080-86c0-82aeec7d9597
# ╟─a2d4e774-6c12-4c2c-a213-44cedc6c1667
# ╟─7ce9332f-3639-465f-bea1-417308914c47
# ╠═b77e9684-fb8a-4d71-9820-f11654508850
# ╠═4ed1a61c-a916-46f2-abb2-f95837db3149
# ╟─bf12a345-7f36-41cf-8567-84819a99e17a
# ╟─359dc820-3ee9-4633-99e4-4e4eb76b6548
# ╟─6b666e20-e9f2-4c3c-adb4-9a556139dc20
# ╠═a09f4eea-50ff-4307-96a9-811ec1148949
# ╠═a9cb62af-091f-472f-892e-c0aca842e5b1
# ╠═7fa6e288-20bd-468f-8fb4-bb50cb4ee548
# ╠═2b41dc34-d3d9-4736-8511-17049f87e604
# ╠═7dff4447-10fa-46a5-a1ac-0a64962449c3
# ╠═e080aaf6-1453-4142-91cc-7ee26dd68f13
# ╟─8a6325b6-8c55-4418-8163-72210a9ef3b4
# ╠═4ae7f11e-728c-47b6-a810-39ce5bccd1ea
# ╠═7b80e86a-3009-42d3-9ea9-8e9e245fecd5
# ╠═103372c5-49aa-44d7-ae79-5af08892c5b1
# ╠═770d537d-dc21-45fa-8ab5-55554201e479
# ╠═146252a2-834d-4ead-9358-f3e520dc54a1
# ╠═ff6fdcad-1196-468a-bfe3-52e406c5decb
# ╠═a4fd9c40-4e99-4f5c-b9b4-6078611407dc
# ╠═89bce87e-b4db-4726-bb9e-e67dab205089
# ╠═403d03b1-e6d4-423a-94b4-05dd6fa37ebc
# ╠═733acbdd-c57f-4e05-a522-be97f7ef2ae6
# ╠═dff95946-ce40-469b-83a4-0c3459840d98
# ╠═6834d85e-271f-4994-932c-919ca5cd36f8
# ╠═30d18139-5c12-4ac1-b196-63bcd90ef5d7
# ╠═da36ec94-fa8a-4927-8b37-756c4a0bdba8
# ╠═a9f16b3c-bbc5-4f0d-84cf-a9ee59c986b4
# ╠═cda2c9cd-935e-4ca0-bd4a-a99caf68b5a7
# ╠═bd40b6dd-3441-4f76-9983-73b6fea43a62
# ╠═c330dc0f-b941-4ece-ba1c-0a8301aa7d33
# ╠═250fcce2-7a9d-417a-8aaa-f407450ffb09
# ╠═7d157865-23f7-46a9-b04f-7e61ce3085e5
# ╠═c31f3bc1-f78d-49e7-99a3-e21fcf2abb2a
# ╠═e18bdab5-458d-4aaa-bbf4-06b07d2071f7
# ╠═e48108d1-1507-4ed5-bb35-6d5b18f83c76
# ╠═901ac58c-24e1-4242-9a80-c1925cb36184
# ╠═ec72a849-17df-40c7-aafc-390083fdfd6e
# ╠═62fbffea-699c-44e8-a1fe-6b957c641a61
# ╠═75ff52ac-8832-4a47-aa56-81d9083348bb
# ╠═19d8a1f4-8bf2-4f18-a8cd-1f1e4d12d143
# ╠═fa193035-ff5c-4334-b65f-695445db4160
# ╠═676bb8c8-5e13-4c49-bed4-62fd92e44779
# ╠═92755d15-ea82-4313-8e67-945d777080c0
# ╠═7b1ad86e-1ca0-4666-8807-32f60b1e6191
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
