# Simultaneous Penalized Quantile Regression Splines

from cmath import tau
from ctypes import Union
import pandas as pd
import numpy as np
import math

# Degree
CONSTANT = 0
LINEAR = 1
QUADRATIC = 2
CUBIC = 3

# Initialize equally spaced knots 
def dim2knots(predictor: np.array, dim: int, degree: int) -> np.array:
    sample_size = len(predictor)
    knot_size = dim-degree+1
    knots = np.zeros(knot_size, dtype = float)
    i = 0.0
    for j in range(knot_size-1):
        i = math.floor(sample_size / (knot_size - 1.0) * j)
        knots[j]= predictor[i]

    knots[knot_size-1] = max(predictor) + 1e-5
    return knots

# find the difference between each consecutive pair of elements of a vector
def pairwiseDiff(arr: np.array) -> np.array:
    n = len(arr)
    diff = []
    for i in range(n-1):
        diff.append(arr[i+1]-arr[i])
    return diff

# For inference on tails
def knots2t(knots: np.array, degree: int = 1) -> np.array :
    d = np.mean(pairwiseDiff(knots))
    min_knot = min(knots)
    max_knot = max(knots)
    n = len(knots)
    t = np.zeros(2*degree+n)
    for j in range(degree):
        t[j] = min_knot - d * (degree-j)
    
    k = j+1
    for j in range(n):
        t[k] = knots[j]
        k += 1

    for j in range(degree):
        t[k] = max_knot + d * (j+1)
        k += 1
    return t

# from R-spqrs
def bsp(x: float, t: np.array, degree: int, j: int) -> float:
    if (degree == 0):
        if ((t[j] <= x) and (x < t[j+1])):
            return 1
        else:
            return 0
    else:
        a, b, c, d = 0,0,0,0
        k = degree + 1 # order
        jd = j+degree
        jpk = j+k
        jp1 = j+1
        c = t[jd]-t[j]
        if (c > 0):
            a = (x-t[j])/c
        d = t[jpk]-t[jp1]
        if (d > 0):
            b = (t[jpk] - x) / (t[jpk] - t[jp1])
    return a * bsp(x, t, degree - 1, j) + b * bsp(x, t, degree - 1, jp1)

# from R-spqrs
def dbsp(x: float, t: np.array, degree: int, j: int, derivative: int) -> float: 
    if (derivative == 0):
      return bsp(x, t, degree, j)
    else:
        a, b, c, d = 0,0,0,0
        c = t[j + degree] - t[j]
        if (c > 0):
            a = degree / c
        k = degree + 1; # k = order
        jp1 = j + 1
        d = t[j + k] - t[jp1]
        if (d > 0):
            b = degree / d
    return a * dbsp(x, t, degree - 1, j, derivative - 1) - b * dbsp(x, t, degree - 1, j + 1, derivative - 1)

# from R-spqrs
def bspline(x: np.array, t: np.array, degree: int = 1, j: int = 0, derivative: int = 0) -> np.array:
    n = len(x)
    b = np.zeros(n)
    k = degree + 1 # order

    if (derivative == 0):
        for i in range(n):
            if ((t[j] <= x[i]) and (x[i] < t[j + k])):
                b[i] = bsp(x[i], t, degree, j)
    else:
        for i in range(n):
            if ((t[j] <= x[i]) and (x[i] < t[j + k])):
                b[i] = dbsp(x[i], t, degree, j, derivative)           
    return b

# find bs coefficient matrix
def bsplines(x: np.array, t: np.array, degree: int = 1, derivative: int = 0) -> np.array:
    n = len(x)
    dim = len(t) - degree - 1
    bm = np.zeros((n, dim), dtype = float)
    for i in range(dim):
        v = bspline(x, t, degree, i, derivative)
        for j in range(n):
            bm[j][i] = v[j]
    return bm

# from R-spqrs
def jump_bsplines(t: np.array, degree: int = 1) -> np.array:
    k = degree + 1 # order
    dim = len(t)-k
    jump =  np.zeros((dim, dim-k), dtype = float)
    x = np.zeros(dim-k+1, dtype = float)
    derivative = np.zeros(dim, dtype= float)
    for j in range(dim-k+1):
        x[j] = 0.5 * (t[j + k - 1] + t[j + k])
    for j in range(dim):
        derivative = bspline(x, t, degree, j, degree)
        for l in range(dim - k):
            jump[j][l] = derivative[l + 1] - derivative[l]
    return jump

# compute all lambdas
def lambdas_all(number_lambdas: int, lambda_max: float, epsilon_lambda: float):
    # compute all lambdas
    # log l_k = log l_1 + (K - k) (log l_K - log l_1)/(K - 1), k = 1, ..., K
    # observe log l_1 = log l_K and log l_K = log l_1
    lambdas_all = np.zeros(number_lambdas, dtype = float)
    lambda_min = epsilon_lambda * lambda_max
    ratio_max_min = 1.0/ epsilon_lambda
    div = number_lambdas - 1
    for idx in range(number_lambdas):
        exponent = idx / div
        lambdas_all[idx] = lambda_min * np.power(ratio_max_min, exponent)
    return lambdas_all

def support_of_vector(v: np.array) -> np.array:
    n = len(v)
    z2n = []
    for i in range(n):
        if v[i] > 1e-6:
            z2n.append(i)
    return z2n

def find_interval_weight(w: np.array, tau: float) -> int:
    # find k satisfies :
    # w[1:(k-1)] < tau * sum(w) <= w[1:k]
    left = 0
    right = len(w)-1
    tau_w_sum = tau * np.sum(w)

    # k = 0
    if (tau_w_sum <= w[0]):
        return 0
    # k != 0
    while (left <= right):
        mid = round((left+right)/2)
        if (np.sum(w[0:mid]) < tau_w_sum):
            left = mid + 1
        elif (np.sum(w[0:mid]) > tau_w_sum):
            right = mid - 1
        else:
            return mid
    return left-1
    
# bubble sort that returns index values
def bubble_order(vec: np.array) -> np.array:
    tmp = 0.0
    n = len(vec)
    clone_vec = vec.copy()
    outvec = np.arange(0, n)
    passes = 0
    while True:
        no_swaps = 0
        for i in range(0, n-1-passes):
            if(clone_vec[i] > clone_vec[i+1]):
                no_swaps += 1
                tmp = clone_vec[i]
                clone_vec[i] = clone_vec[i + 1]
                clone_vec[i + 1] = tmp
                itmp = outvec[i]
                outvec[i] = outvec[i+1]
                outvec[i+1] = itmp
        if (no_swaps == 0):
            break
        passes += 1
    return outvec

def subsetNumVec(x: np.array, index: np.array) -> np.array:
    n = len(index)
    # index = [i-1 for i in index]
    out = np.zeros(n)
    for i in range(n):
        out[i] = x[index[i]]
    return out

def subsetIntVec(x: np.array, index: np.array) -> np.array:
    n = len(index)
    # index = [i-1 for i in index]
    out = np.zeros(n, dtype = int)
    for i in range(n):
        out[i] = x[index[i]]
    return out

def find_interval(y: np.array, z: float) -> int:
    left = 0
    right = len(y)-1
    while (left <= right):
        mid = math.ceil((left+right)/2)
        # If z greater than y[middle], ignore left half.
        if (y[mid] < z):
            left = mid + 1
        # If x is smaller, ignore right half.
        elif (y[mid] > z):
            right = mid - 1
        else:
        # z is present at middle
            return mid
    return left - 1

def type_forward(target: float, tau_penalty: float, lam: float, a: float, c: float):
    if (c <= target):
        return lam * a * tau_penalty
    else:
        return -lam * a * (1 - tau_penalty)

def type_backward(target: float, tau_penalty: float, lam: float, a: float, c: float):
    if (c <= target):
        return lam * a * (1 - tau_penalty)
    else:
        return -lam * a * tau_penalty

def find_slope_v(v: np.array, w: np.array, tau:float, tau_penalty: float, lam: float, a: np.array, c: np.array, i:int, pen_type: np.array) ->bool:
    P = len(c)
    slope = 0.0
    target = v[i]

    if (i == -1):
        target = -math.inf
        slope = -tau * sum(w)
    else:
        slope = sum(w[0:i]) - tau * sum(w)
    
    for p in range(P):
        if (pen_type[p] == 1):
            slope += type_forward(target, tau_penalty, lam, a[p], c[p])
        else:
            slope += type_backward(target, tau_penalty, lam, a[p], c[p])
    
    return True if slope >= 0 else False

def find_slope_c(v: np.array, w: np.array, tau:float, tau_penalty: float, lam: float, a: np.array, c: np.array, l:int, pen_type: np.array) ->bool:
    P = len(c)
    slope = 0.0
    i_star_l = find_interval(v, c[l])

    if (i_star_l == -1):
        slope = -tau * sum(w)
    else:
        slope = sum(w[0:i_star_l]) - tau * sum(w)
    
    for p in range(P):
        if (pen_type[p] == 1):
            slope += type_forward(c[l], tau_penalty, lam, a[p], c[p])
        else:
            slope += type_backward(c[l], tau_penalty, lam, a[p], c[p])
    
    return True if slope >= 0 else False


def find_solution(v: np.array, w:np.array, tau: float, tau_penalty: float, lam: float, a: np.array, c: np.array, pen_type: np.array) ->float:
    m = len(v)
    P = len(c)
    order_c = bubble_order(c)
    a = subsetNumVec(a, order_c)
    pen_type = subsetIntVec(pen_type, order_c)
    c.sort()
    i_star = np.zeros(P, dtype = int)
    for p in range(P):
        i_star[p] = find_interval(v, c[p])
    for p in range(P):
        if (not find_slope_v(v, w, tau, tau_penalty, lam, a, c, i_star[p], pen_type)):
            if (find_slope_c(v, w, tau, tau_penalty, lam, a, c, p, pen_type)):
                return c[p]
    left = 0
    right = m - 1
    while (left < right):
        middle = math.floor((right + left) / 2)
        if (not find_slope_v(v, w, tau, tau_penalty, lam, a, c, middle, pen_type)):
            left = middle + 1
        else:
            right = middle
    return v[left-1]
    
def diff(a: np.array, b:np.array) -> np.array:
    assert(len(a) == len(b))
    diff = np.zeros(len(a))
    for i in range(len(a)):
        diff[i] = a[i]-b[i]
    return diff

def check(u: float, tau: float) -> float:
    if (u < 0):
        return (tau-1.0)*u
    else:
        return tau * u

def Check(v: np.array, tau: float) -> float:
    risk = 0.0
    for i in range(len(v)):
        risk += check(v[i], tau)
    return risk       

# simultaneous estimation of quantile regression functions
# using B-splines And Total Variation Penalty
def spqrs(
    predictor: np.array,
    response: np.array,
    tau: np.array,
    degree: int,
    dimension: int = 10,
    tau_penalty: float = 0.5,
    num_lambdas: int = 4,
    lambda_max: float = 1e+2,
    epsilon_lambda: float = 1e-10,
    max_iter: int = 500,
    epsilon_iterations: float = 1e-6,
    non_cross_constr: bool = False):

    sample_size = len(response)
    
    order = degree + 1
    num_taus = len(tau)
    dimension_vec = np.array([dimension for i in range(num_taus)])

    # init_knots
    knot_list = []
    t_list = []
    coefficient_list = []
    residual_list = []

    for i in range(num_taus):
        knots = dim2knots(predictor, dimension_vec[i], degree)
        knot_list.append(np.array(knots))
        t_list.append(np.array(knots2t(knots, degree)))
        initial_coef = np.zeros(dimension_vec[i], dtype = float)
        coefficient_list.append(np.array(initial_coef))
        rsp = response.copy()
        residual_list.append(np.array(rsp))

    # all lambdas
    lambdas = lambdas_all(num_lambdas, lambda_max, epsilon_lambda)
    
    # local variables
    results = []
    b_j = np.zeros(sample_size)
    v_j = np.zeros(sample_size)
    w_j = np.zeros(sample_size)
    residual_j = np.zeros(sample_size)

    # criterion
    aic_vector = np.zeros(num_lambdas)
    bic_vector = np.zeros(num_lambdas)
    R_vec = np.zeros(num_taus)
    R_lambda_vec = np.zeros(num_taus)
    lam, R, R_lam, store_R_lam, sol_candidate = 0.0, 0.0, 0.0,  0.0,  0.0
    i, iter, j, k, l, q = 0,0,0,0,0,0
    number_pruning = 10

    number_penalty_vec = np.zeros(num_taus, dtype = int)
    for i in range(num_taus):
        number_penalty_vec[i] = dimension_vec[i] - order

    tilde_B_km1 = bsplines(predictor, knot_list[0], degree, 0)
    tilde_B_k = tilde_B_km1
    # fit
    for lam_idx in range(num_lambdas):
        lam = lambdas[lam_idx]
        ## fit a spqrs corresponding to the lambda
        for iter in range(max_iter):
            ## module taus
            for tau_idx in range(num_taus):
                basis = bsplines(predictor, t_list[tau_idx], degree, 0)
                jump = jump_bsplines(t_list[tau_idx], degree)
                supports = []
                
                for j in range(dimension_vec[tau_idx]):
                    supports.append(np.array(support_of_vector(basis[:,j])))
                
                if non_cross_constr:
                    if tau_idx > 0:
                        sum_knots = np.union1d(knot_list[tau_idx-1], knot_list[tau_idx])
                        sum_knots = np.unique(sum_knots)
                        sum_knots.sort()
                        tilde_B_km1 = bsplines(sum_knots, t_list[tau_idx-1], degree, 0)
                        tilde_B_k = bsplines(sum_knots, t_list[tau_idx], degree, 0)
                                 
                for j in range(dimension_vec[tau_idx]):
                    b_j = basis[:,j]
                    if supports[j].size == 0:
                        print("wtf")
                        continue
                    for k in range(len(supports[j])):
                        residual_list[tau_idx][supports[j][k]] += b_j[supports[j][k]] * coefficient_list[tau_idx][j]
                    
                    w_j = b_j[supports[j]]
                    residual_j = residual_list[tau_idx][supports[j]]
                    v_j = residual_j/w_j
                    order_v = bubble_order(v_j)
                    w_j = subsetNumVec(w_j, order_v)
                    # breakpoint()
                    v_j.sort()
                    
                    if (number_penalty_vec[tau_idx] == 0):
                        # weighted quantile
                        q = find_interval_weight(w_j, tau[tau_idx])
                        sol_candidate = v_j[q]
                        
                        if (tau_idx == 0):
                            # print("a")
                            coefficient_list[tau_idx][j] = sol_candidate
                        if (not non_cross_constr):
                            # print("b")
                            coefficient_list[tau_idx][j] = sol_candidate
                            # breakpoint()
                    else: 
                        # compute a, c
                        # compute a, d vector for penalty term
                        nonzero_size = 0
                        row_jump = np.array(jump[j])
                        for l in range(len(row_jump)):
                            if (abs(row_jump[l]) > 1e-6):
                                nonzero_size += 1
                        # memory for a, c
                        c = np.zeros(nonzero_size)
                        a = np.zeros(nonzero_size)
                        nonzero_index = np.zeros(nonzero_size, dtype = int)
                        ss = 0
                        for l in range(len(row_jump)):
                            if (abs(row_jump[l]) > 1e-6):
                                a[ss] = row_jump[l]
                                nonzero_index[ss] = l
                                ss += 1
                        # compute c
                        for k in range(nonzero_size):
                            c[k] = coefficient_list[tau_idx][j] - sum(jump[:, nonzero_index[k]] * coefficient_list[tau_idx]) / a[k]
                        # abs a after compute c
                        a = abs(a)
                        # pen_type
                        pen_type = np.zeros(nonzero_size, dtype = int)
                        for k in range(nonzero_size):
                            if (jump[j][nonzero_index[k]] > 0):
                                pen_type[k] = 1
                        
                        sol_candidate = find_solution(v_j, w_j, tau[tau_idx], tau_penalty, lam, a, c, pen_type)

                        if (tau_idx == 0):
                            # print("k")
                            coefficient_list[tau_idx][j] = sol_candidate
                            # breakpoint()
                        if (not non_cross_constr):
                            # print("y")
                            coefficient_list[tau_idx][j] = sol_candidate
                            # breakpoint()
                    # recompute residual
                    for k in range(len(supports[j])):
                        residual_list[tau_idx][supports[j][k]] -= b_j[supports[j][k]] * coefficient_list[tau_idx][j]
            
                # module prune  
                if (iter <= number_pruning):
                    if (number_penalty_vec[tau_idx] > 0):
                       
                        penalty = np.zeros(number_penalty_vec[tau_idx])
                        for k in range(number_penalty_vec[tau_idx]):
                            penalty[k] = sum(jump[:,k] * coefficient_list[tau_idx])
                        penalty_check = np.zeros(number_penalty_vec[tau_idx], dtype = int)
                        penalty_check = [(1 if abs(penal) < 1e-6 else penal) for penal in penalty]
                        
                        if (sum(penalty_check) > 0):
                            #  recompute
                            prune_coef = np.zeros(dimension_vec[tau_idx], dtype = int)
                            prune_knots = np.zeros(number_penalty_vec[tau_idx]+2, dtype = int)
                            for k in range(number_penalty_vec[tau_idx]):
                                if abs(penalty[k]) < 1e-6:
                                    prune_coef[k] = 1
                                    prune_knots[k+1] = 1
                            # zero_coef_index = [i for i,d in enumerate(prune_coef) if d==0]
                            # coefficient_list[tau_idx] = [coefficient_list[tau_idx][ii] for ii in zero_coef_index]
                            coefficient_list[tau_idx] = coefficient_list[tau_idx][prune_coef == 0]
                            # zero_knot_index = [i for i,d in enumerate(prune_knots) if d==0]
                            # knot_list[tau_idx] = [knot_list[tau_idx][jj] for jj in zero_knot_index]
                       
                            knot_list[tau_idx] = knot_list[tau_idx][prune_knots == 0]
                            dimension_vec[tau_idx] = len(coefficient_list[tau_idx])

                            # fitted values
                            fitted_values = np.zeros(sample_size)
                            number_penalty_vec[tau_idx] = dimension_vec[tau_idx] - order
                            t_list[tau_idx] = knots2t(knot_list[tau_idx], degree)
                            basis = bsplines(predictor, t_list[tau_idx], degree, 0)
                            if number_penalty_vec[tau_idx] > 0:
                                jump = jump_bsplines(t_list[tau_idx], degree)
                            for j in range(dimension_vec[tau_idx]):
                                supports[j] = support_of_vector(basis[:, j])
                                b_j = basis[:, j]
                                for k in range(len(supports[j])):
                                    fitted_values[supports[j][k]] += b_j[supports[j][k]] * coefficient_list[tau_idx][j]
                            
                            residual_list[tau_idx] = diff(response, fitted_values)

                # module fit
                R_vec[tau_idx] = Check(residual_list[tau_idx], tau[tau_idx])
                R_lambda_vec[tau_idx] = R_vec[tau_idx]
                for k in range(number_penalty_vec[tau_idx]):
                    R_lambda_vec[tau_idx] += lam * check(sum(jump[:,k] * coefficient_list[tau_idx]), tau_penalty)
                
            # module check_convergence
            R = sum(R_lambda_vec)
            R_lam = sum(R_lambda_vec)
            if (iter > 10):
                if (abs(R_lam - store_R_lam) < epsilon_iterations):
                    break
            store_R_lam = R_lam
        
        for k in range(num_taus):
            aic_vector[lam_idx] += sample_size * np.log(R_vec[k]) + 2 * dimension_vec[k]
            bic_vector[lam_idx] += sample_size * np.log(R_vec[k]) + np.log(sample_size) * dimension_vec[k]
        
        results.append({"knots_list": knot_list, "coefficient_list": coefficient_list, "dimension_vec" : dimension_vec})

    results.append({"bic_vector" : bic_vector, "aic_vector" : aic_vector, "lambdas" :lambdas})
    return results


            

    
    