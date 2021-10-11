using Base: Bool
using LinearAlgebra
using SparseArrays

# functions

function unity(i,n)
    E = spzeros(Bool,n,n)
    E[i] = true
    return E
end

# xs = [rand(3,4) for _ in 1:5]
# reduce((x,y) -> cat(x,y,dims=3), A)
# idea: https://github.com/JuliaLang/julia/pull/37196
# reduce(cat(dims=4), [ones(2,3) for _ in 1:5]) |> size 
#`reduce(vcat, [rand(2,3) for _ = 1:4]): A[i][a,b] -> A[(i-1)*len_a + a, b]`
#which is the same as restructuring to` B[i,a][b] via B = [A[i][a,:] for i = 1:dim_i, a = 1:dim_a] `
#and then taking `reduce(hcat,B'[:]')'`
#`reduce(hcat, [rand(2,3) for _ = 1:4]): A[i][a,b] -> A[a, (i-1)*len_b + b]` 

function partial_trace(dim_1, dim_2, traced_system)
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

function ptr(ρ, dim_1, traced_system = 1)
    # get partial trace superoperator
    N,M = size(ρ)
    dim_2 = N÷dim_1
    PT = partial_trace(dim_1, dim_2, traced_system) 
    new_dim = traced_system == 1 ? dim_2 : dim_1
    # apply superoperator to vectorized density matrix and reshape to new dimensions
    σ = reshape(PT * ρ[:], new_dim, new_dim)
    return σ
end

### apply split
function factorize(A, dim_1, dim_2)
    σ = prt(A, dim_1, dim_2, 2)
    ρ = prt(A, dim_1, dim_2, 1)
    A_tensor_prod = kron(σ, ρ)
    rho_c = A - A_tensor_prod
    return A_tensor_prod, rho_c
end

### density matrix

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


function choi2kraus(C)
    dim_1,dim_2 = size(C)
    N= Int64(sqrt(dim_1))
    K = [zeros(ComplexF64, N,N) for _=1:N^2]
    ew,ev = eigen(C)
    for k = 1:N^2
        K[k] = reshape(conj.(ev[:,k]),N,N) .* sqrt(abs(ew[k]))
        if ew[k] < 0
            print("Warning, negative eigenvalue, k = " * string(k))
        end
    end
    return K
end

function kraus2choi(K)
    dim1,dim2 = size(K[1])
    C = zeros(ComplexF64, dim1^2,dim1^2)
    for k = 1:length(K)
        C += conj.(K[k])[:] * transpose(K[k][:])
    end
    return C
end
### ======================== functions end =======================

dim_1 = 2
dim_2 = 3
m = 7
n = 3

A = rand(m,m)
B = rand(n,n)


C = kron(A,B)

A_sup = kron(A[:], I(n^2))
C_sup = choi(C,m)

sum(abs.(A_sup * B[:] - C_sup[:]))

A_sup * B[:] - C[:]
reshape(A_sup * B[:],n*m,n*m) - C_sup
sum(abs.(ans[:]))


# create superoperator without applying Choi on the right hand side

G = reshape(permutedims(reshape(kron(A[:], I(n^2)), (n,n,m,m,n^2)), (1,3,2,4,5)), n^2*m^2,n^2)

reshape(G*B[:], m*n, m*n)
C
sum(abs.(G * B[:] - C[:]))

kron_fac_B = reshape(permutedims(reshape(kron(I(m^2), B[:]), (n,n,m,m,m^2)), (1,3,2,4,5)), n^2*m^2, m^2)
norm(reshape(kron_fac_B * A[:], n*m, n*m) - C)


function I_rect(n1,n2)
    if n1 >= n2
        temp = I(n1)
        temp = temp[:,1:n2]
    else
        temp = I(n2)
        temp = temp[1:n1,:]
    end
    return temp
end


m1 = 3
m2 = 4

n1 = 5
n2 = 6

A = rand(m1,m2)
B = rand(n1,n2)

C = kron(A,B)

#G = reshape(permutedims(reshape(kron(A[:], I(n1*n2)), (n1,n2,m1,m2,n1*n2)), (1,3,2,4,5)), n1*n2*m1*m2,n1*n2)

S_left = reshape(permutedims(reshape(kron(I(n1*n2), A[:]), (m1,m2,n1,n2,n1*n2)), (3,1,4,2,5)), prod([m1,m2,n1,n2]), n1*n2)
norm(S_left * B[:] - C[:])

S_right = reshape(permutedims(reshape(kron(B[:], I(m1*m2)), (m1,m2,n1,n2,m1*m2)), (3,1,4,2,5)), prod([m1,m2,n1,n2]), m1*m2)
norm(S_right * A[:] - C[:])


norm(kron(A[:], I(n1*n2)) * B[:] - choi_rect(C, n1,m1, n2, m2)[:])

norm(kron(I(m1*m2), B[:]) * A[:] - choi_rect(C,n1,m1,n2,m2)[:])

norm(kron(A'[:], I(n1*n2)) * B'[:] - C'[:])

# elegant version to show right hand side as a Choi matrix for first and second factor:
norm(kron(I(n1*n2) ,A[:]) * B[:] - choi_rect(C, n1, m1, n2, m2)'[:])
norm(kron(B[:], I(m1*m2)) * A[:] - choi_rect(C, n1, m1, n2, m2)'[:])


# check embeddings

A = rand(2)
B = rand(3)
C = rand(4)
D = rand(5)

Q = kron(A, B')
kron(kron(A, B'), ones(4,5))

kron(kron(A,B'), kron(C,D')) - kron(kron(A, B'), I(4)) * kron(I(4), kron(C,D'))

Q = kron(A, C')
kron(kron(A, ones(3)'), kron(C, ones(5)'))- kron(Q'[:], ones(3*5)')

Q = kron(A, D')
kron(kron(A, ones(3)'), kron(ones(4), D'))-kron(kron(ones(3)', Q), ones(4))

Q = kron(B, C')
kron(kron(ones(2), B'), kron(C, ones(5)')) - kron(kron(ones(2),Q'), ones(5)')

Q = kron(B, D')
kron(kron(ones(2), B'), kron(ones(4), D'))-kron(ones(2*4), Q[:]')

# multiplication table for nxn unity base matrices

n = 4
A = zeros(Int64,n^2,n^2)
function unity(i,n)
    E = spzeros(Bool,n,n)
    E[i] = true
    return E
end

for i = 1:n^2, j=1:n^2
    L = findall((unity(i,n)*unity(j,n))[:] .> 0)
    if isempty(L)
        A[i,j] = 0
    else
        A[i,j] = L[1]
    end
end
A




# try to check \sum_{i != j} a_ii a_jj - abs(a_ij)^2 == sum_{i \neq j} \lambda_i lambda_j

A = density_matrix(10,0);
#Diagonal(real(A))
ew,ev = eigen(A);
L1 = [ A[i,i]*A[j,j] - abs(A[i,j])^2 for i = 1:5, j = 1:5];
L2 = [ew[i] * ew[j] * (i != j) for i = 1:5, j=1:5];

sum(real(L1) .< 0)
real(ew)'


# can every density matrix be represented as the sum of tensor products of valid density matrices?

A = density_matrix(4)
B = density_matrix(3)

ew, ev = eigen(kron(A,B))
sum(ew .< 0)


a = rand(4)

# the Kraus decomposition for initially correlated dynamical maps:
A6 = density_matrix(6)
A2 = density_matrix(2)
A3 = density_matrix(3)
A4 = density_matrix(4)

A = density_matrix(9)

n_runs = 80

tr_weight = zeros(n_runs)

A =  kron(A3, A4)
for k = 1:n_runs
    T1, C1 = factorize(A./tr(A), 2,6)

    A, M1 = split(C1)
    tr_weight[k] = tr(A)
end
tr_weight
sum(cumprod(tr_weight))


# entangled state

A = [1 0 0 1; 0 0 0 0; 0 0 0 0; 1 0 0 1]/2
t,c = factorize(A/tr(A),2,2)
p,m = split(c)

ω = tr(p)
p = p/tr(p)
m = m/tr(m)
round)
function split(rho)
    ew, ev = eigen(rho)
    rho_p = Hermitian(ev[:,ew .> 0] * Diagonal(ew[ew.> 0]) * ev[:,ew .> 0]')
    rho_m = Hermitian(ev[:,ew .< 0] * Diagonal(-ew[ew.< 0]) * ev[:,ew .< 0]')
    return rho_p, rho_m
end






P1 = partial_trace(2,3,1)
P2 = partial_trace(2,3,2)

A1 = reshape(P2*A[:],2,2)
A2 = reshape(P1*A[:],3,3)

rho_c = A - kron(A1,A2)
rho_c = Hermitian(rho_c)

ew,ev = eigen(rho_c)

rho_p = Hermitian(ev[:,ew .> 0] * Diagonal(ew[ew.> 0]) * ev[:,ew .> 0]')
rho_m = Hermitian(ev[:,ew .< 0] * Diagonal(-ew[ew.< 0]) * ev[:,ew .< 0]')

# now decompose both again, maybe it does converge



A = rand(4,4)
B = rand(4,4)


kron(A,B)' - kron(A', B')



# Choi to Kraus representation

A = density_matrix(16) * 4


ew, ev = eigen(A)
K = [zeros(ComplexF64,4,4) for _ in 1:16]
for k = 1:16
    K[k] = reshape(ev[:,k], 4,4) .* sqrt(ew[k])
end

Test = zeros(ComplexF64,4,4)
for k=1:16
    Test = Test + K[k] * K[k]'
end
Test
tr(Test)
B = density_matrix(4)
B2 = zeros(ComplexF64,4,4)
for k = 1:16
    B2 += K[k] * B * K[k]'
end
tr(B)
tr(B2)




A = rand(4,4)
A = A 

exp(-im*A) * exp(-im*A)'

# general way to construct a valid quantum operation:
# find A \in C^{d^2 x d^2}
# with A^\dag = A and A>0
# then A2 = A + (I - P_13 A) \kron I/d is a valid Choi matrix

# create choi matrix directly fulfilling the trace condition

A = density_matrix(16)*4
P13 = partial_trace(4,4,2) #tracing out the second system - flip to switch to colexicographic
P24 = partial_trace(4,4,1)

A2 = A + kron(I(4) - reshape(P13*A[:], 4,4), I(4)/4)


ew, ev = eigen(A2); ew
round(sum(abs.(ew)) - tr(A2), digits=14)

reshape(P13*A2[:], 4,4)
# unital map: Tr_24 yields I(N)
A3 = A2 + kron(I(4)/4, I(4) - reshape(P24*A[:],4,4))

round.(reshape(P13*A3[:], 4,4), digits = 14)

round.(reshape(P24*A3[:], 4,4), digits = 14)

# get Kraus operators from A2 from eigenvectors 
#(complex conjugation can be ignored since C is hermitian):
K = [zeros(ComplexF64, 4,4) for _=1:16]
for k = 1:16
    K[k] = reshape(ev[:,k],4,4) * sqrt(ew[k])

end
K = choi2kraus(A2)
id_test = zeros(ComplexF64,4,4)
for k = 1:16
    id_test += K[k]' * K[k]
end

id_test



C = kraus2choi(K)
sum(abs.(C - (A2)))
C
A2

#Ginibre matrix NxN
#G = randn(N,N)
# 1/(2 pi )^(N^2/2) * exp( -1/2 Tr(G * transpose(G)))

N= 1000
G = randn(N,N)
1/(2*pi)^(N^2/2) * exp(- 1/2 * tr(G * transpose(G)))


# create random Kraus operators from large unitary matrix
N = 4
A = rand(N^2*N, N^2*N)
ew, ev = eigen(Hermitian(A))
K = [zeros(ComplexF64, 4,4) for _=1:16]
for k=1:16
    K[k] = ev[1:4, (1:4) .+ (k-1)*4]'
end

# get the choi matrix from the Kraus operators
C = zeros(ComplexF64, 16,16)

for k = 1:16
    C = C + K[k]'[:] * K[k]'[:]'
end
ew, ev = eigen(C);
tr(C) - sum(ew)
sum(abs.(C - C'))

# get linear superoperator
D = choi(C,4)


p1 = partial_trace(4,4,1)

Test  = zeros(4,4)
reshape(sum(p1, dims = 1),4,4)


A = randn(12,16) + randn(12,16) * 0.1*im
C_ = A'*A/tr(A'*A)*4
w,v = eigen(C_); w

#C_ = density_matrix(16) * 4
C = (C_ + kron(I(4) - ptr(C_, 4, 2), I(4)/4)) 
tr(C)
ew, ev = eigen(C);ew

ptr(C,4,2)
reshape(P13 * C[:],4,4)


# Kraus operators of partial transpose

#transpose operator
T = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

C = choi(T,2)
ew, ev = eigen(C)

K = choi2kraus(C)
A = density_matrix(2)
temp = zeros(ComplexF64,2,2)
for k = 1:4
    temp += K[k] * A * K[k]'
    if k == 1
        temp *=-1
    end
end
temp - A
A
Id_test  = zeros(ComplexF64,2,2)
for k = 1:4
 Id_test += K[k]' * K[k]
end
Id_test

K[1]' * K[1]

A = rand(4,4)

B = rand(3,3)


# tensor product as superoperator

# L(X) = A ⊗ X

A = rand(2,2)
X = rand(3,3)
# aim:
L * X[:] == kron(A,X)[:]

CL = kron(A[:], I(9))
L = reshape(permutedims(reshape(CL, 3,3,2,2,9),[1,3,2,4,5]), 36,9)
L * X[:]
kron(A,X)[:]

L2 = kron(kron(A,I(3))[:], I(3)[:]')
L


i = 16

n = [2,3,2,3]

a = [(i-1) % n[1]+1,
((i-1) % prod(n[1:2]) ÷ n[1]) + 1,
(i-1) % prod(n[1:3]) ÷ prod(n[1:2]) + 1,
(i-1) ÷ prod(n[1:3]) + 1]

a[1] + n[1] * (a[2]-1 + n[2] * (a[3] - 1 + n[3] * (a[4]-1)))


function choi_index(i, n)
    return  (i.-1) .% n[1] .+ 1 .+
    n[1] .* ((i.-1) .% prod(n[1:3]) .÷ prod(n[1:2]) .+ 
    n[2] .* ((i.-1) .% prod(n[1:2]) .÷ n[1] .+
    n[3] .* ((i.-1) .÷ prod(n[1:3]))))
end
T = reshape(1:36,2,3,2,3)
