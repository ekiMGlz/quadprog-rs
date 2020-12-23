use nalgebra::{DMatrix, DVector, QR, SymmetricEigen};
use std::collections::{BTreeMap, HashSet};
use crate::base::*;

// struct for problem data after presolving
#[derive(Debug)]
pub(crate) struct RawQP {
    pub(crate) hess: DMatrix<f64>, // problem.hess.symmetric()
    pub(crate) c: DVector<f64>,
    pub(crate) a: DMatrix<f64>,    // extended ineq matrix [-A; I; -I]
    pub(crate) b: DVector<f64>,    // extended ineq rhs [-b; lb; -ub]
    pub(crate) a_eq: DMatrix<f64>, // A_eq, full rank, no full zero rows
    pub(crate) b_eq: DVector<f64>, // b_eq to match A_eq changes
    pub(crate) dims: Dims,
    pub(crate) x0: Option<DVector<f64>>,
    pub(crate) options: crate::base::QPOptions,
}

#[derive(Debug)]
pub(crate) struct Dims {
    pub(crate) n: usize,
    pub(crate) m: usize,   // = m_ineq + m_lb + m_ub
    pub(crate) m_eq: usize,
    pub(crate) m_ineq: usize,
    pub(crate) m_lb: usize,
    pub(crate) m_ub: usize
}

pub(crate) fn presolve(problem: &ConvexQP) -> Result<RawQP, QPError>{

    let eps = 1e-8;

    // Check dimensions
    let (n, hess, c) = {
        let n1 = problem.c.len();
        let (n2, n3) = problem.hess.shape();
        if n2 == n3 {
            if n1 == n2{
                if let Some((i, _)) = problem.hess.iter().enumerate().find(|&(_, &x)| {!x.is_finite()}) {
                    let row = i % n2;
                    let col = i / n2;

                    return Err(QPError::Undefined{matrix: "hess".to_string(), row, col})
                }

                if let Some((i, _)) = problem.c.iter().enumerate().find(|&(_, &x)| {!x.is_finite()}) {
                    let row = i;

                    return Err(QPError::Undefined{matrix: "c".to_string(), row, col: 0})
                }

                // Maybe change to cholesky fact to check for convexity
                let eig_decom = SymmetricEigen::new(problem.hess.symmetric_part());
                if let Some(x) = eig_decom.eigenvalues.iter().find(|&&x| {x < -eps}){
                    return Err(QPError::Infeasible(format!("Non-convex problem, negative eigenvalue ({}) found in hessian matrix.", x)))
                }

                // Maybe send eigs numerically zero to 0.0

                (n1, eig_decom.recompose(), problem.c.clone())
            }else{
                return Err(QPError::DimensionMismatch{matrix: "c".to_string(), dims: (n1, 1), expected_dims:(n2, 1)});
            }
        }else{
            return Err(QPError::DimensionMismatch{matrix: "hess".to_string(), dims: (n2, n3), expected_dims:(n2, n2)});
        }
    };

    let (m_eq, a_eq, b_eq) = if let Some(vec) = &problem.b_eq {
        let m1 = vec.len();
        if let Some(matrix) = &problem.a_eq {
            let (m2, n2) = matrix.shape();
            if m1 == m2{
                if n == n2{
                    if let Some((i, _)) = matrix.iter().enumerate().find(|&(_, &x)| {!x.is_finite()}) {
                        let row = i % m2;
                        let col = i / m2;

                        return Err(QPError::Undefined{matrix: "a_eq".to_string(), row, col})
                    }

                    if let Some((i, _)) = vec.iter().enumerate().find(|&(_, &x)| {!x.is_finite()}) {
                        let row = i;

                        return Err(QPError::Undefined{matrix: "b_eq".to_string(), row, col: 0})
                    }

                    // Correct dims
                    let mut a_eq = matrix.clone();
                    let mut b_eq = vec.clone();

                    // Check and remove zero rows
                    for i in (0..m1).rev(){
                        if a_eq.row(i).amax() < eps {
                            if b_eq[i].abs() < eps {
                                a_eq = a_eq.remove_row(i);
                                b_eq = b_eq.remove_row(i);
                            }else{
                                return Err(QPError::Infeasible(format!("Equality constraint {} is infeasible, zero row with non-zero rhs ({}).", i, b_eq[i])));
                            }

                        }
                    }

                    // Check a_eq for full rank
                    let m = a_eq.nrows();

                    if m > n {
                        return Err(QPError::Infeasible(format!("a_eq has too many non-zero rows ({}), must have less than the number of variables ({})", m, n)));
                    }

                    let r = QR::new(a_eq.clone()).r();
                    if r.row(m - 1).amax() < eps {
                        // QR factorization has a 0 row in R matrix with m <=n, therefore A_eq is rank deficient
                        return Err(QPError::RankDeficient("Equality constraints are rank deficient, check for redundant constraints.".to_string()))
                    }

                    (m, a_eq, b_eq)

                }else{
                    return Err(QPError::DimensionMismatch{matrix: "a_eq".to_string(), dims: (m2, n2), expected_dims:(m2, n)});
                }
            }else{
                return Err(QPError::DimensionMismatch{matrix: "b_eq".to_string(), dims: (m1, 1), expected_dims:(m2, 1)});
            }
        }else {
            if vec.len() > 0 {
                return Err(QPError::DimensionMismatch{matrix: "a_eq".to_string(), dims: (0, n), expected_dims:(vec.len(), n)});
            }
            (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
        }
    }else {
        if let Some(matrix) = &problem.a_eq {
            if matrix.nrows() > 0 {
                return Err(QPError::DimensionMismatch{matrix: "b_eq".to_string(), dims: (0, 1), expected_dims: (matrix.nrows(), 1)});
            }
        };
        (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
    };

    let (mut bounds, mut m_lb, mut m_ub) = if let Some(map) = &problem.bounds {
        // verify & count bounds

        let mut m_lb = 0;
        let mut m_ub = 0;

        let bounds: BTreeMap<_, _> = map.iter().filter_map(|(&k, &v)| {
            if k >= n{
                None
            }else {
                match v {
                    (None, None) => None,
                    (None, Some(x)) if x.is_infinite() & x.is_sign_positive() => None,
                    (Some(x), None) if x.is_infinite() & x.is_sign_negative() => None,
                    (Some(x), Some(y)) if x.is_infinite() & x.is_sign_negative() & y.is_infinite() & y.is_sign_positive()=> None,
                    _ => Some((k, v))
                }
            }
        }).collect();

        for (&k, &v) in bounds.iter(){
            match v {
                (Some(x), _) if x.is_infinite() & x.is_sign_positive() => return Err(QPError::Infeasible(format!("Variable {} has +∞ lower bound.", k))),
                (_, Some(x)) if x.is_infinite() & x.is_sign_negative() => return Err(QPError::Infeasible(format!("Variable {} has -∞ upper bound.", k))),
                (Some(lb), Some(ub)) if lb > ub => return Err(QPError::Infeasible(format!("Variable {} has inconsistent bounds [{}, {}].", k, lb, ub))),
                (Some(lb), Some(ub)) => {if lb.is_finite() {m_lb += 1}; if ub.is_finite() {m_ub += 1};},
                (Some(_), None) => {m_lb += 1;},
                _ => {m_ub += 1;},
            }
        }

        (bounds, m_lb, m_ub)

    }else{
        (BTreeMap::new(), 0, 0)
    };

    let (m_ineq, mut a, mut b) = if let Some(vec) = &problem.b {
        let m1 = vec.len();
        if let Some(matrix) = &problem.a {
            let (m2, n2) = matrix.shape();
            if m1 == m2{
                if n == n2{
                    if let Some((i, _)) = matrix.iter().enumerate().find(|&(_, &x)| {!x.is_finite()}) {
                        let row = i % m2;
                        let col = i / m2;

                        return Err(QPError::Undefined{matrix: "a".to_string(), row, col})
                    }

                    if let Some((i, _)) = vec.iter().enumerate().find(|&(_, &x)| {!x.is_finite()}) {
                        let row = i;

                        return Err(QPError::Undefined{matrix: "b".to_string(), row, col: 0})
                    }

                    // Correct dims
                    let mut a = matrix.clone();
                    let mut b = vec.clone();

                    // Check a for 0 rows or singleton rows
                    for i in (0..m1).rev(){
                        let row = a.row(i);
                        let mut row_iter = row.iter().enumerate().filter(|(_, x)| {x.abs() > eps});
                        match row_iter.next() {
                            None => {
                                // 0 nonzero elements
                                if b[i] > -eps {
                                    a = a.remove_row(i);
                                    b = b.remove_row(i);
                                }else{
                                    return Err(crate::base::QPError::Infeasible(format!("Inequality constraint {} is infeasible, zero row with negative rhs ({}).", i, b[i])));
                                }
                            },
                            Some((j, x)) => {
                                if let None = row_iter.next() {
                                    // Exactly 1 nonzero element. replace ineq with bound
                                    let bound = b[i] / x;

                                    // Update bound, checking feasibility
                                    match bounds.get(&j) {

                                        Some((None, Some(ub))) => {
                                            let ub = *ub;
                                            if x.is_sign_positive() {
                                                if bound < ub {
                                                    // tighter bound found, update
                                                    bounds.insert(j, (None, Some(bound)));
                                                }
                                                // looser bound, ignore
                                            }else{
                                                if bound <= ub {
                                                    // lower bound found, update
                                                    bounds.insert(j, (Some(bound), Some(ub)));
                                                    m_lb += 1;
                                                }else {
                                                    // inconsistent bounds
                                                    return Err(crate::base::QPError::Infeasible(format!("Inequality constraint {} contradicts bounds for variable {}.", i, j)))
                                                }
                                            }
                                        },
                                        Some((Some(lb), None)) => {
                                            let lb = *lb;

                                            if x.is_sign_positive() {
                                                if bound >= lb {
                                                    // upper bound found, update
                                                    bounds.insert(j, (Some(lb), Some(bound)));
                                                    m_ub += 1;
                                                }else{
                                                    // inconsistent bounds
                                                    return Err(crate::base::QPError::Infeasible(format!("Inequality constraint {} contradicts bounds for variable {}.", i, j)))
                                                }

                                            }else{
                                                if bound > lb {
                                                    // tighter bound found, update
                                                    bounds.insert(j, (Some(bound), None));
                                                }
                                                // lloser bound, ignore
                                            }
                                        },
                                        Some((Some(lb), Some(ub))) => {
                                            let ub = *ub;
                                            let lb = *lb;

                                            if x.is_sign_positive() {
                                                if bound < ub {
                                                    if bound >= lb {
                                                        // tighter upper bound found
                                                        bounds.insert(j, (Some(lb), Some(bound)));
                                                    }else {
                                                        // inconsistent bound
                                                        return Err(crate::base::QPError::Infeasible(format!("Inequality constraint {} contradicts bounds for variable {}.", i, j)))
                                                    }
                                                }
                                                // looser upper bound
                                            }else{
                                                if bound > lb {
                                                    if bound <= ub {
                                                        // tighter lower bound found
                                                        bounds.insert(j, (Some(bound), Some(ub)));
                                                    }else {
                                                        // inconsistent bound
                                                        return Err(crate::base::QPError::Infeasible(format!("Inequality constraint {} contradicts bounds for variable {}.", i, j)))
                                                    }
                                                }
                                                // looser lower bound
                                            }
                                        },
                                        _ => {
                                            if bound.is_finite() {
                                                // Variable is not bound, add the bound
                                                if x.is_sign_positive() {
                                                    bounds.insert(j, (None, Some(bound)));
                                                    m_ub += 1;
                                                } else {
                                                    bounds.insert(j, (Some(bound), None));
                                                    m_lb += 1;
                                                }
                                            }else{
                                                if bound.is_sign_negative() & x.is_sign_positive() {return Err(QPError::Infeasible(format!("Inequality constraint {} shows a -∞ upper bound.", i)))};
                                                if bound.is_sign_positive() & x.is_sign_negative() {return Err(QPError::Infeasible(format!("Inequality constraint {} shows a +∞ lower bound.", i)))};
                                            }


                                        },

                                    }

                                    // Remove the inequality
                                    a = a.remove_row(i);
                                    b = b.remove_row(i);

                                }
                                // else More than 1 nonzero element
                            },
                        }
                    }

                    let m_ineq = a.nrows();
                    (m_ineq, -a, -b)

                }else{
                    return Err(QPError::DimensionMismatch{matrix: "a".to_string(), dims: (m2, n2), expected_dims:(m2, n)});
                }
            }else{
                return Err(QPError::DimensionMismatch{matrix: "b".to_string(), dims: (m1, 1), expected_dims:(m2, 1)});
            }
        }else {
            if vec.len() > 0 {

                return Err(QPError::DimensionMismatch{matrix: "a".to_string(), dims: (0, n), expected_dims:(vec.len(), n)});
            }
            (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
        }
    }else {
        if let Some(matrix) = &problem.a {
            if matrix.nrows() > 0 {
                return Err(QPError::DimensionMismatch{matrix: "b".to_string(), dims: (0, 1), expected_dims:(matrix.nrows(), 1)});
            }
        };
        (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
    };

    // Insert rows for bounds
    a = a.insert_rows(m_ineq, m_lb + m_ub, 0.0);
    b = b.insert_rows(m_ineq, m_lb + m_ub, 0.0);

    for (i, (j, v)) in bounds.iter().filter(|(_, v)| {matches!(v, (Some(x), _) if x.is_finite())}).enumerate(){
        a[(m_ineq + i, *j)] = 1.0;
        b[m_ineq + i] = v.0.unwrap();
    }

    for (i, (j, v)) in bounds.iter().filter(|(_, v)| {matches!(v, (_, Some(x)) if x.is_finite())}).enumerate(){
        a[(m_ineq + m_lb + i, *j)] = -1.0;
        b[m_ineq + m_lb + i] = -v.1.unwrap();
    }

    Ok(RawQP {
        hess,
        c,
        a,
        b,
        a_eq,
        b_eq,
        dims: Dims{n, m: m_ineq + m_lb + m_ub, m_eq, m_ineq, m_lb, m_ub},
        x0: problem.x0.clone(),
        options: problem.options.clone()
    })
}

pub(crate) fn interior_point_method(problem: RawQP) -> Result<QPSolution, QPError>{

    // Unpack rawqp
    let Dims{n, m, m_eq, m_ineq, m_lb, m_ub} = problem.dims;
    let c = problem.c;
    let hess = problem.hess;
    let a = problem.a;
    let a_eq = problem.a_eq;
    let b_eq = problem.b_eq;
    let b = problem.b;
    let options = problem.options;

    // Build the Jacobi matrix of the system
    let mut newton_raphson_sys = DMatrix::<f64>::zeros(n + m_eq + 2*m, n + m_eq + 2*m);

    newton_raphson_sys.slice_mut((0, 0), (n, n)).copy_from(&hess);
    newton_raphson_sys.slice_mut((n, 0), (m_eq, n)).copy_from(&a_eq);
    newton_raphson_sys.slice_mut((n + m_eq, 0), (m, n)).copy_from(&a);

    let r = &-newton_raphson_sys.slice((n, 0), (m_eq + m, n)).transpose();
    newton_raphson_sys.slice_mut((0, n), (n, m_eq + m)).copy_from(r);
    newton_raphson_sys.slice_mut((n + m_eq, n + m_eq + m), (m, m)).copy_from(&-DMatrix::<f64>::identity(m, m));
    newton_raphson_sys.slice_mut((n + m_eq + m, n + m_eq), (m, m)).copy_from(&DMatrix::<f64>::identity(m, m));
    newton_raphson_sys.slice_mut((n + m_eq + m, n + m_eq + m), (m, m)).copy_from(&DMatrix::<f64>::identity(m, m));


    // Init values for multipliers and slacks
    let mut eq_multipliers = DVector::<f64>::zeros(m_eq);
    let mut ineq_multipliers = DVector::<f64>::repeat(m, 1.0);
    let mut slacks = ineq_multipliers.clone();

    // Check for an initial solution, generate one otherwise
    let mut x0 = if let Some(x) = &problem.x0{
        x.clone()
    }else{
        let x0 = DVector::<f64>::repeat(n, 1.0);

        // for i in 0..n {
        //     if i < m_lb && i < m_ub {
        //         x0[i] = 0.5 * (problem.lb.as_ref().unwrap()[i] + problem.ub.as_ref().unwrap()[i]);
        //     }else if i < m_lb {
        //         x0[i] = 1.1 * problem.lb.as_ref().unwrap()[i] ;
        //     }else if i < m_ub {
        //         x0[i] = 1.1 * problem.ub.as_ref().unwrap()[i];
        //     }
        // }

        // let mut residues = DVector::<f64>::repeat(n + m_eq + 2 * m_extended, 1.0);
        // let hess = newton_raphson_sys.slice((0, 0), (n, n));
        // let a_eq = newton_raphson_sys.slice((n, 0), (m_eq, n));
        // let a = newton_raphson_sys.slice((n + m_eq, 0), (m_extended, n));

        // let ones = DVector::<f64>::repeat(m_extended, 1.0);

        // let r_dual = &(hess*&x0 + &c - a.transpose()*&ones);
        // let r_eq = &(a_eq*&x0 - &b_eq);
        // let r_ineq = &(a*&x0 - &b - &ones);
        // residues.rows_mut(0, n).copy_from(r_dual);
        // residues.rows_mut(n, m_eq).copy_from(r_eq);
        // residues.rows_mut(n + m_eq, m_extended).copy_from(r_ineq);

        // let delta = QR::new(newton_raphson_sys.clone()).solve(&-residues).unwrap();

        // let mut alpha_k: f64 = 1.0;
        // for i in 0..m_extended{
        //     if delta[n + m_eq + i] < 0.0 {
        //         alpha_k = alpha_k.min(-ineq_multipliers[i]/delta[n + m_eq + i])
        //     }
        //     if delta[n + m_eq + m_extended + i] < 0.0 {
        //         alpha_k = alpha_k.min(-slacks[i]/delta[n + m_eq + m_extended + i])
        //     }
        // }

        // alpha_k = alpha_k * 0.95;

        // x0 += delta.rows(0, n).map(|x|{x * alpha_k});
        // eq_multipliers += delta.rows(n, m_eq).map(|x|{x * alpha_k});
        // ineq_multipliers += delta.rows(n + m_eq, m_extended).map(|x|{x * alpha_k});
        // slacks += delta.rows(n + m_eq + m_extended, m_extended).map(|x|{x * alpha_k});

        x0
    };

    let mut residues = DVector::<f64>::zeros(n + m_eq + 2 * m);
    let mut first_order_cond: f64 = 0.0;
    let tol_scaling = [1.0, hess.norm(), c.norm(), a.norm(), a_eq.norm(), b.norm(), b_eq.norm()].iter().copied().fold(f64::NAN, f64::max);

    for iterations in 0..options.max_iterations{

        // Check the first order conditions

        let r_dual = &(&hess*&x0 + &c - &a_eq.transpose()*&eq_multipliers - &a.transpose()*&ineq_multipliers);
        let r_eq = &(&a_eq*&x0 - &b_eq);
        let r_ineq = &(&a*&x0 - &b - &slacks);
        let mut r_s = slacks.component_mul(&ineq_multipliers);

        // Stopping conditions
        if r_eq.lp_norm(1) + r_ineq.lp_norm(1) <= tol_scaling*options.con_tol && r_dual.lp_norm(1) <=tol_scaling * options.fun_tol{
            let soln = QPSolution{
                fval: 0.5*x0.dot(&(hess*&x0)) + x0.dot(&c),
                x: x0,
                ineq_multipliers: if m_ineq > 0 {Some(ineq_multipliers.rows(0, m_ineq).clone_owned())} else {None},
                eq_multipliers: if m_eq > 0 {Some(eq_multipliers)} else {None},
                lb_multipliers: if m_lb > 0 {Some(ineq_multipliers.rows(m_ineq, m_lb).clone_owned())} else {None},
                ub_multipliers: if m_ub > 0 {Some(ineq_multipliers.rows(m_ineq + m_lb, m_ub).clone_owned())} else {None},
                iterations: iterations,
                first_order_cond: first_order_cond,
            };

            return Ok(soln)
        }

        residues.rows_mut(0, n).copy_from(r_dual);
        residues.rows_mut(n, m_eq).copy_from(r_eq);
        residues.rows_mut(n + m_eq, m).copy_from(r_ineq);
        residues.rows_mut(n + m_eq + m, m).copy_from(&r_s);

        first_order_cond = residues.amax();

        // Update Jacobi matrix
        newton_raphson_sys.slice_mut((n + m_eq + m, n + m_eq), (m, m)).copy_from(&DMatrix::<f64>::from_diagonal(&slacks));
        newton_raphson_sys.slice_mut((n + m_eq + m, n + m_eq + m), (m, m)).copy_from(&DMatrix::<f64>::from_diagonal(&ineq_multipliers));

        // QR factorization of the Jacobi matrix
        let qr = QR::new(newton_raphson_sys.clone());

        let affine_step = match qr.solve(&-&residues){
            Some(x) => x,
            None => return Err(QPError::Infeasible(format!("Interior point method: Jacobi matrix is singular.")))
        };

        // Cut the step for feasibility
        let mut alpha: f64 = 1.0;
        for i in 0..m{
            if affine_step[n + m_eq + i] < 0.0 {
                alpha = alpha.min(-ineq_multipliers[i]/affine_step[n + m_eq + i])
            }
            if affine_step[n + m_eq + m + i] < 0.0 {
                alpha = alpha.min(-slacks[i]/affine_step[n + m_eq + m + i])
            }
        }

        let delta_z_aff = affine_step.rows(n + m_eq, m).map(|x|{x*alpha});
        let delta_s_aff = affine_step.rows(n + m_eq + m, m).map(|x|{x*alpha});

        // Calculate centrality parameter
        let duality_measure = slacks.dot(&ineq_multipliers)/(m as f64);
        let affine_duality_measure = slacks.zip_map(&delta_s_aff, |x, y|{x + y}).dot(&ineq_multipliers.zip_map(&delta_z_aff, |x, y|{x + y})) / (m as f64);
        let centrality_param = (affine_duality_measure/duality_measure).powi(3) * duality_measure;

        // Calculate Mehrotra predictor-corrictor step
        r_s = r_s + delta_s_aff.zip_map(&delta_z_aff, |x, y|{ x*y - centrality_param });
        residues.rows_mut(n + m_eq + m, m).copy_from(&r_s);

        let step = match qr.solve(&-&residues){
            Some(x) => x,
            None => return Err(QPError::Infeasible(format!("Interior point method: Jacobi matrix is singular.")))
        };

        // Cut the step for feasibility
        alpha = 1.0;
        for i in 0..m{
            if step[n + m_eq + i] < 0.0 {
                alpha = alpha.min(-ineq_multipliers[i]/step[n + m_eq + i])
            }
            if step[n + m_eq + m + i] < 0.0 {
                alpha = alpha.min(-slacks[i]/step[n + m_eq + m + i])
            }
        }


        // Update step
        x0 += step.rows(0, n).map(|x|{x * alpha});
        eq_multipliers += step.rows(n, m_eq).map(|x|{x * alpha});
        ineq_multipliers += step.rows(n + m_eq, m).map(|x|{x * alpha});
        slacks += step.rows(n + m_eq + m, m).map(|x|{x * alpha});

    }

    Err(QPError::MaxIterationsReached{algorithm: QPAlgorithm::InteriorPoint,
    fval: 0.5*x0.dot(&(hess*&x0)) + x0.dot(&c),
    last_soln: x0,
    first_order_cond})
}

pub(crate) fn active_set_method(problem: RawQP) -> Result<QPSolution, QPError> {
    let eps = 1e-8;
    
    // Unpack rawqp
    let Dims{n, m, m_eq, m_ineq, m_lb, m_ub} = problem.dims;
    let c = problem.c;
    let hess = problem.hess;
    let a = problem.a;
    let a_eq = problem.a_eq;
    let b_eq = problem.b_eq;
    let b = problem.b;

    let phase1_hess = DMatrix::<f64>::zeros(n + 1, n + 1);
    let mut phase1_c = DVector::<f64>::zeros(n + 1);
    phase1_c[0] = 1.0;
    let phase1_a_eq = a_eq.clone().insert_column(0, 0.0);
    let phase1_a = a.clone().insert_row(m, 0.0).insert_column(0, 1.0);
    let phase1_b_eq = b_eq.clone();

    let rho = a_eq.amax().max(a.amax()) * problem.options.con_tol;
    let phase1_b = b.clone().insert_row(m, -rho);


    let mut init_soln = DVector::<f64>::zeros(n + 1);
    if m_eq > 0 {
        let (q, r) = full_qr(&a_eq.transpose());
        let x = q.columns(0, m_eq)*forward_substitution(&r.transpose(), &b_eq);

        init_soln[0] = (&a*&x-&b).max() + 1.0;
        init_soln.rows_mut(1, n).copy_from(&x);

    }else{
        init_soln[0] = b.max() + 1.0;
    }

    let phase1 = RawQP{
        hess: phase1_hess,
        c: phase1_c,
        a: phase1_a,
        b: phase1_b,
        a_eq: phase1_a_eq,
        b_eq: phase1_b_eq,
        dims: Dims{
            n: n + 1,
            m: m + 1,
            m_eq,
            m_ineq,
            m_lb: 1,
            m_ub: 0
        },
        x0: Some(init_soln),
        options: QPOptions{
            algorithm: QPAlgorithm::InteriorPoint,
            max_iterations: 100,
            opt_tol: 1e-8,
            step_tol: 1e-8,
            fun_tol: 1e-8,
            con_tol: 1e-8,
            verbose: false,
        }
    };

    let soln = interior_point_method(phase1)?;
    
    let mut x0 = soln.x.rows(1, n).clone_owned();
    let mut active_constraints = a_eq.transpose();
    let ineq_constraints = &a.transpose();


    let mut working_set: Vec<usize> = Vec::new();
    let mut inactive_set: HashSet<usize> = (0..m).collect();
    let mut rank = m_eq;

    for iter in 0..problem.options.max_iterations{
        

        let (q, r) = full_qr(&active_constraints);
        let null_space = q.columns(rank, n - rank);
        let grad = -(&hess*&x0 + &c);
        let p = match QR::new(null_space.transpose()*&hess*&null_space).solve(&(null_space.transpose()*&grad)){
            Some(x) => &null_space * x,
            None => return Err(QPError::Infeasible(format!("Active set method: ZQZ is singular")))
        };

        if p.amax() < eps {
            if rank == 0{

                let eq_multipliers = if m_eq > 0 {Some(DVector::<f64>::zeros(m_eq))} else{ None };
                let ineq_multipliers = if m_ineq > 0 {Some(DVector::<f64>::zeros(m_ineq))} else{ None };
                let lb_multipliers = if m_lb > 0 {Some(DVector::<f64>::zeros(m_lb))} else{ None };
                let ub_multipliers = if m_ub > 0 {Some(DVector::<f64>::zeros(m_ub))} else{ None };

                return Ok(QPSolution{
                    fval: 0.5*x0.dot(&(&hess*&x0)) + x0.dot(&c),
                    x: x0,
                    eq_multipliers,
                    ineq_multipliers,
                    lb_multipliers,
                    ub_multipliers,
                    iterations: iter,
                    first_order_cond: 0.0,
                })
            }else{

                let multipliers = backwards_substitution(&r, &(q.columns(0, rank).transpose() * -&grad));
                
                if rank == m_eq || multipliers.rows(m_eq, rank - m_eq).min() > -eps {
                    let eq_multipliers = Some(multipliers.rows(0, m_eq).clone_owned());
                    let mut ineq_multipliers = DVector::<f64>::zeros(m_ineq);
                    let mut lb_multipliers = DVector::<f64>::zeros(m_lb);
                    let mut ub_multipliers = DVector::<f64>::zeros(m_ub);

                    for (i, mu) in working_set.iter().copied().zip(multipliers.iter().skip(m_eq).copied()) {
                        if i < m_ineq {
                            ineq_multipliers[i] = mu;
                        }else if i < m_ineq + m_lb {
                            lb_multipliers[i - m_ineq] = mu;
                        }else{
                            ub_multipliers[i - m_ineq - m_lb] = mu;
                        }
                    }

                    let ineq_multipliers = if m_ineq > 0 {Some(ineq_multipliers)} else{ None };
                    let lb_multipliers = if m_lb > 0 {Some(lb_multipliers)} else{ None };
                    let ub_multipliers = if m_ub > 0 {Some(ub_multipliers)} else{ None };

                    return Ok(QPSolution{
                        fval: 0.5*x0.dot(&(&hess*&x0)) + x0.dot(&c),
                        x: x0,
                        eq_multipliers,
                        ineq_multipliers,
                        lb_multipliers,
                        ub_multipliers,
                        iterations: iter,
                        first_order_cond: 0.0,
                    })
                }else{
                    let j = multipliers.rows(m_eq, rank - m_eq).imin();
                    active_constraints = active_constraints.remove_column(m_eq + j);
                    let constraint_index = working_set.remove(j);
                    inactive_set.insert(constraint_index);
                    rank -=1 ;
                }
            }
        }else{

            let mut alpha: f64 = 1.0;
            let mut imin = None;

            for i in inactive_set.iter() {
                let row = ineq_constraints.column(*i);

                let a_dot_p = row.dot(&p);

                if a_dot_p < -eps {
                    let slack = b[*i] - row.dot(&x0);
                    
                    if slack.abs() < eps{
                        alpha = 0.0;
                        imin = Some(*i);
                        break;
                    }

                    let cut = slack/a_dot_p;
                    if alpha > cut {
                        alpha = cut;
                        imin = Some(*i);
                    }
                }
            }

            if let Some(i) = imin {
                working_set.push(i);
                inactive_set.remove(&i);
                active_constraints = active_constraints.insert_column(rank, 0.0);
                active_constraints.column_mut(rank).copy_from(&ineq_constraints.column(i));
                rank += 1;
            }


            x0 = x0 + alpha * p;
        }

    }


    Err(QPError::MaxIterationsReached{algorithm: QPAlgorithm::ActiveSet,
    fval: 0.5*x0.dot(&(hess*&x0)) + x0.dot(&c),
    last_soln: x0,
    first_order_cond: 0.0})
}


fn full_qr(matrix: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>){

    let (m, n) = matrix.shape();

    if m==0 || n==0 {
        ( DMatrix::<f64>::identity(m, m), matrix.clone())
    }else{
        let mut q = DMatrix::<f64>::identity(m, m);
        let mut v = -matrix.column(0).clone();

        // Check if v is (x, 0, 0, ..., 0)T, otherwise use a reflector
        if  v.rows(1, m - 1).amax() > 1e-6{
            v[0] += v.norm();

            let v_dot_v = v.dot(&v);

            for i in 0..m {
                for j in i..m {
                    q[(i, j)] -= 2.0 * v[i] * v[j] / v_dot_v;
                    q[(j, i)] = q[(i, j)];
                }
            }
        };



        let mut r = &q*matrix;

        for col in 1..n.min(m) {
            let vec_size = m - col;
            let mut v = -r.slice((col, col), (vec_size, 1)).clone();

            if v.rows(1, vec_size - 1).amax() > 1e-6{
                v[0] += v.norm();
                let v_dot_v = v.dot(&v);

                let mut h = DMatrix::<f64>::identity(m, m);
                let mut reflector = h.slice_mut((col, col), (vec_size, vec_size));

                for i in 0..vec_size{
                    for j in i..vec_size{
                        reflector[(i, j)] -= 2.0 * v[i] * v[j] / v_dot_v;
                        reflector[(j, i)] = reflector[(i, j)];
                    }
                }

                r = &h*r;
                q = &h*q;
            };

        }

        (q.transpose(), r)
    }
}

fn forward_substitution(lower: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64>{
    // solves a square Lx=b system
    let n = b.len();
    let mut x = DVector::<f64>::zeros(n);

    for i in 0..n{
        let mut sum = b[i];
        for j in 0..i{
            sum -= lower[(i, j)] * x[j];
        }
        x[i] = sum / lower[(i,i)];
    }

    x
}

fn backwards_substitution(upper: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64>{
    // solves a square Ux=b system
    let n = b.len();
    let mut x = DVector::<f64>::zeros(n);

    for i in (0..n).rev(){
        let mut sum = b[i];
        for j in i + 1..n{
            sum -= upper[(i, j)] * x[j];
        }
        x[i] = sum / upper[(i,i)];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn qr_rank() {
        let mat = DMatrix::<f64>::from_vec(2, 3, vec!(1.0, -1.0, 3.0, -3.0, 5.0, -5.0));
        let qr = QR::new(mat);
        let r = qr.r();
        assert!(r.row(1).amax() < 1e-8);
    }

    #[test]
    fn ineq_tests(){
        assert!(Some(5.0) > None);
        assert!(!(Some(5.0) < None));
        assert!(f64::NEG_INFINITY < -100000000.0);
        assert!(f64::INFINITY > 100000000.0);
        assert!(!(f64::INFINITY < f64::INFINITY));

    }

    #[test]
    fn presolve_hess_c() {
        let file = File::open("qp/presolve/hess_c/minimal_problem.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();
        let raw_problem = presolve(&problem).unwrap();

        assert_eq!(raw_problem.dims.n, 3);
        assert_eq!(raw_problem.dims.m, 0);
        assert_eq!(raw_problem.dims.m_eq, 0);
        assert_eq!(raw_problem.dims.m_ineq, 0);
        assert_eq!(raw_problem.dims.m_lb, 0);
        assert_eq!(raw_problem.dims.m_ub, 0);
        assert_eq!(raw_problem.a.shape(), (0, 3));
        assert_eq!(raw_problem.a_eq.shape(), (0, 3));
        assert_eq!(raw_problem.b.len(), 0);
        assert_eq!(raw_problem.b_eq.len(), 0);
    }

    #[test]
    fn presolve_hess_inf() {
        let file = File::open("qp/presolve/hess_c/inf_hess.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Undefined{matrix, row, col} = error {
            assert_eq!(matrix, "hess".to_string());
            assert_eq!(row, 1);
            assert_eq!(col, 2);
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_c_nan() {
        let file = File::open("qp/presolve/hess_c/nan_c.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Undefined{matrix, row, col} = error {
            assert_eq!(matrix, "c".to_string());
            assert_eq!(row, 3);
            assert_eq!(col, 0);
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_rectangular_hess() {
        let file = File::open("qp/presolve/hess_c/rectangular_hess.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();

        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "hess".to_string());
            assert_eq!(dims, (3, 2));
            assert_eq!(expected_dims, (3, 3));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }

    }

    #[test]
    fn presolve_rectangular_hess2() {
        let file = File::open("qp/presolve/hess_c/rectangular_hess2.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "hess".to_string());
            assert_eq!(dims, (2, 3));
            assert_eq!(expected_dims, (2, 2));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_dims_hess_c() {
        let file = File::open("qp/presolve/hess_c/dims_hess_c.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "c".to_string());
            assert_eq!(dims, (3, 1));
            assert_eq!(expected_dims, (2, 1));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq(){
        let file = File::open("qp/presolve/eq/base.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        assert_eq!(raw.a_eq.shape(), (2, 4));
        assert_eq!(raw.a_eq.nrows(), raw.b_eq.nrows());

    }

    #[test]
    fn presolve_eq_inf_a(){
        let file = File::open("qp/presolve/eq/inf_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Undefined{matrix, row, col} = error {
            assert_eq!(matrix, "a_eq".to_string());
            assert_eq!(row, 2);
            assert_eq!(col, 3);
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_nan_b() {
        let file = File::open("qp/presolve/eq/nan_b.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Undefined{matrix, row, col} = error {
            assert_eq!(matrix, "b_eq".to_string());
            assert_eq!(row, 2);
            assert_eq!(col, 0);
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_oversized_a() {
        let file = File::open("qp/presolve/eq/oversized_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Infeasible(msg) = error {
            assert!(msg.contains("a_eq has too many non-zero rows (5)"))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_wrong_cols_a() {
        let file = File::open("qp/presolve/eq/wrong_cols_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "a_eq".to_string());
            assert_eq!(dims, (3, 3));
            assert_eq!(expected_dims, (3, 4));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_a_no_b() {
        let file = File::open("qp/presolve/eq/a_no_b.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "b_eq".to_string());
            assert_eq!(dims, (0, 1));
            assert_eq!(expected_dims, (3, 1));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_b_no_a() {
        let file = File::open("qp/presolve/eq/b_no_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "a_eq".to_string());
            assert_eq!(dims, (0, 4));
            assert_eq!(expected_dims, (3, 4));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_diff_b_a() {
        let file = File::open("qp/presolve/eq/diff_b_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "b_eq".to_string());
            assert_eq!(dims, (2, 1));
            assert_eq!(expected_dims, (3, 1));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_zero_rows_err() {
        let file = File::open("qp/presolve/eq/zero_rows_err.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Infeasible(msg) = error {
            assert!(msg.contains("Equality constraint 2 is infeasible, zero row with non-zero rhs (1)"))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_rank_def() {
        let file = File::open("qp/presolve/eq/rank_def.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::RankDeficient(msg) = error {
            assert!(msg.contains("Equality constraints are rank deficient, check for redundant constraints."))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_eq_rank_def_zero_rows() {
        let file = File::open("qp/presolve/eq/rank_def_zero_rows.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::RankDeficient(msg) = error {
            assert!(msg.contains("Equality constraints are rank deficient, check for redundant constraints."))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_bounds() {
        let file = File::open("qp/presolve/bounds/base.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        // Verify eq constraints
        assert_eq!(raw.a_eq.shape(), (2, 4));
        assert_eq!(raw.a_eq.nrows(), raw.b_eq.nrows());

        // Verify bounds
        assert_eq!(raw.dims.m_lb, 2);
        assert_eq!(raw.dims.m_ub, 2);

        let mut a = DMatrix::<f64>::zeros(4, 4);
        a[(0, 1)] = 1.0;
        a[(1, 2)] = 1.0;
        a[(2, 2)] = -1.0;
        a[(3, 3)] = -1.0;
        let b = DVector::<f64>::from_vec(vec!(0.0,0.0,-1.0,-2.0));

        assert_eq!(raw.a, a);
        assert_eq!(raw.b, b);
    }

    #[test]
    fn presolve_bounds_inf_lbound() {
        let file = File::open("qp/presolve/bounds/inf_lbound.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Infeasible(msg) = error {
            assert!(msg.contains("Variable 0 has +∞ lower bound."))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_bounds_ninf_ubound() {
        let file = File::open("qp/presolve/bounds/neg_inf_ubound.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Infeasible(msg) = error {
            assert!(msg.contains("Variable 0 has -∞ upper bound."))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_bounds_inconsistent() {
        let file = File::open("qp/presolve/bounds/inconsistent_bounds.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Infeasible(msg) = error {
            assert!(msg.contains("Variable 0 has inconsistent bounds [1, 0]."))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_bounds_redundant() {
        let file = File::open("qp/presolve/bounds/redundant_bounds.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        assert_eq!(raw.dims.m_lb, 0);
        assert_eq!(raw.dims.m_ub, 0);
        assert_eq!(raw.a.shape(), (0, 4));
        assert_eq!(raw.b.shape(), (0, 1));
    }

    #[test]
    fn presolve_bounds_extra() {
        let file = File::open("qp/presolve/bounds/extra_bounds.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        assert_eq!(raw.dims.m_lb, 0);
        assert_eq!(raw.dims.m_ub, 0);
        assert_eq!(raw.a.shape(), (0, 4));
        assert_eq!(raw.b.shape(), (0, 1));
    }

    #[test]
    fn presolve_ineq() {
        let file = File::open("qp/presolve/ineq/base.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        // Verify eq constraints
        assert_eq!(raw.a_eq.shape(), (2, 4));
        assert_eq!(raw.a_eq.nrows(), raw.b_eq.nrows());

        // Verify bounds
        assert_eq!(raw.dims.m_lb, 2);
        assert_eq!(raw.dims.m_ub, 2);

        // Verify ineq
        assert_eq!(raw.dims.m_ineq, 2);
        assert_eq!(raw.dims.m, 6);

        let mut a = DMatrix::<f64>::zeros(6, 4);
        a.row_mut(0).fill(-1.0);
        a.row_mut(1).fill(1.0);
        a[(2, 1)] = 1.0;
        a[(3, 2)] = 1.0;
        a[(4, 2)] = -1.0;
        a[(5, 3)] = -1.0;
        let b = DVector::<f64>::from_vec(vec!(-10.0,-10.0,0.0,0.0,-1.0,-2.0));

        assert_eq!(raw.a, a);
        assert_eq!(raw.b, b);
    }

    #[test]
    fn presolve_ineq_inf_a(){
        let file = File::open("qp/presolve/ineq/inf_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Undefined{matrix, row, col} = error {
            assert_eq!(matrix, "a".to_string());
            assert_eq!(row, 0);
            assert_eq!(col, 0);
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_nan_b() {
        let file = File::open("qp/presolve/ineq/nan_b.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Undefined{matrix, row, col} = error {
            assert_eq!(matrix, "b".to_string());
            assert_eq!(row, 1);
            assert_eq!(col, 0);
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_zero_row() {
        let file = File::open("qp/presolve/ineq/zero_rows.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        // Verify eq constraints
        assert_eq!(raw.a_eq.shape(), (2, 4));
        assert_eq!(raw.a_eq.nrows(), raw.b_eq.nrows());

        // Verify bounds
        assert_eq!(raw.dims.m_lb, 2);
        assert_eq!(raw.dims.m_ub, 2);

        // Verify ineq
        assert_eq!(raw.dims.m_ineq, 2);
        assert_eq!(raw.dims.m, 6);

        let mut a = DMatrix::<f64>::zeros(6, 4);
        a.row_mut(0).fill(-1.0);
        a.row_mut(1).fill(1.0);
        a[(2, 1)] = 1.0;
        a[(3, 2)] = 1.0;
        a[(4, 2)] = -1.0;
        a[(5, 3)] = -1.0;
        let b = DVector::<f64>::from_vec(vec!(-10.0,-10.0,0.0,0.0,-1.0,-2.0));

        assert_eq!(raw.a, a);
        assert_eq!(raw.b, b);
    }

    #[test]
    fn presolve_ineq_zero_rows_err() {
        let file = File::open("qp/presolve/ineq/zero_rows_err.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Infeasible(msg) = error {
            assert!(msg.contains("Inequality constraint 2 is infeasible, zero row with negative rhs (-10)."))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_wrong_cols_a() {
        let file = File::open("qp/presolve/ineq/wrong_cols_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "a".to_string());
            assert_eq!(dims, (2, 3));
            assert_eq!(expected_dims, (2, 4));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_a_no_b() {
        let file = File::open("qp/presolve/ineq/a_no_b.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "b".to_string());
            assert_eq!(dims, (0, 1));
            assert_eq!(expected_dims, (2, 1));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_b_no_a() {
        let file = File::open("qp/presolve/ineq/b_no_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "a".to_string());
            assert_eq!(dims, (0, 4));
            assert_eq!(expected_dims, (2, 4));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_diff_b_a() {
        let file = File::open("qp/presolve/ineq/diff_b_a.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::DimensionMismatch{matrix, dims, expected_dims} = error {
            assert_eq!(matrix, "b".to_string());
            assert_eq!(dims, (3, 1));
            assert_eq!(expected_dims, (2, 1));
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_singleton_lb() {
        let file = File::open("qp/presolve/ineq/new_lb.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        // Verify bounds
        assert_eq!(raw.dims.m_lb, 2);
        assert_eq!(raw.dims.m_ub, 1);

        // Verify ineq
        assert_eq!(raw.dims.m_ineq, 1);
        assert_eq!(raw.dims.m, 4);

        let mut a = DMatrix::<f64>::zeros(4, 4);
        a.row_mut(0).fill(-1.0);
        a[(1, 0)] = 1.0;
        a[(2, 1)] = 1.0;
        a[(3, 0)] = -1.0;
        let b = DVector::<f64>::from_vec(vec!(-10.0,1.0,1.0,-2.0));

        assert_eq!(raw.a, a);
        assert_eq!(raw.b, b);
    }

    #[test]
    fn presolve_ineq_singleton_ub() {
        let file = File::open("qp/presolve/ineq/new_ub.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        // Verify bounds
        assert_eq!(raw.dims.m_lb, 1);
        assert_eq!(raw.dims.m_ub, 2);

        // Verify ineq
        assert_eq!(raw.dims.m_ineq, 1);
        assert_eq!(raw.dims.m, 4);

        let mut a = DMatrix::<f64>::zeros(4, 4);
        a.row_mut(0).fill(-1.0);
        a[(1, 0)] = 1.0;
        a[(2, 0)] = -1.0;
        a[(3, 1)] = -1.0;
        let b = DVector::<f64>::from_vec(vec!(-10.0,0.0,-2.0,-3.0));

        assert_eq!(raw.a, a);
        assert_eq!(raw.b, b);
    }

    #[test]
    fn presolve_ineq_singleton_lb_err(){
        let file = File::open("qp/presolve/ineq/inconsistent_lb.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Infeasible(msg) = error {
            assert!(msg.contains("Inequality constraint 0 contradicts bounds for variable 0"))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_singleton_ub_err(){
        let file = File::open("qp/presolve/ineq/inconsistent_ub.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let error = presolve(&problem).expect_err("Expected error from presolve");

        if let QPError::Infeasible(msg) = error {
            assert!(msg.contains("Inequality constraint 0 contradicts bounds for variable 0"))
        }else{
            panic!("Wrong type of error from presolve: \"{}\"", error);
        }
    }

    #[test]
    fn presolve_ineq_redundant() {
        let file = File::open("qp/presolve/ineq/redundant_bounds.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();
        let raw = presolve(&problem).unwrap();

        // Verify bounds
        assert_eq!(raw.dims.m_lb, 1);
        assert_eq!(raw.dims.m_ub, 1);

        // Verify ineq
        assert_eq!(raw.dims.m_ineq, 1);
        assert_eq!(raw.dims.m, 3);

        let mut a = DMatrix::<f64>::zeros(3, 4);
        a.row_mut(0).fill(-1.0);
        a[(1, 0)] = 1.0;
        a[(2, 0)] = -1.0;
        let b = DVector::<f64>::from_vec(vec!(-10.0,0.0,-2.0));

        assert_eq!(raw.a, a);
        assert_eq!(raw.b, b);
    }


}
