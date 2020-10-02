use nalgebra::{DMatrix, DVector, QR};
use std::collections::BTreeMap;
use crate::base::{QPError, ConvexQP};

// struct for problem data after presolving
#[derive(Debug)]
pub(crate) struct RawQP {
    hess: DMatrix<f64>, // problem.hess.symmetric()
    c: DVector<f64>,
    a: DMatrix<f64>,    // extended ineq matrix [-A; I; -I]
    b: DVector<f64>,    // extended ineq rhs [-b; lb; -ub]
    a_eq: DMatrix<f64>, // A_eq, full rank, no full zero rows
    b_eq: DVector<f64>, // b_eq to match A_eq changes
    dims: (usize, usize, usize, usize, usize, usize), //(n, m, m_eq, m_ineq, m_lb, m_ub)
    x0: Option<DVector<f64>>,
    options: crate::base::QPOptions,
}

pub(crate) fn presolve(problem: &ConvexQP) -> Result<RawQP, QPError>{

    let numerical_zero = 1e-8;

    // Check dimensions
    let (n, hess, c) = {
        let n1 = problem.c.len();
        let (n2, n3) = problem.hess.shape();
        if n2 == n3 {
            if n1 == n2{
                // TODO: Check hess convexity (?)
                (n1, problem.hess.symmetric_part(), problem.c.clone())
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
            if m2 > n2 {
                return Err(QPError::RankDeficient(format!("a_eq has {} rows, but only {} cols, therefore a_eq does not have full row rank", m2, n2)));
            }
            if m1 == m2{
                if n == n2{
                    // Correct dims
                    let mut a_eq = matrix.clone();
                    let mut b_eq = vec.clone();

                    // Check and remove zero rows
                    for i in (0..m1).rev(){
                        if a_eq.row(i).amax() < numerical_zero {
                            if b_eq[i].abs() < numerical_zero {
                                a_eq = a_eq.remove_row(i);
                                b_eq = b_eq.remove_row(i);
                            }else{
                                return Err(crate::base::QPError::Infeasible(format!("{}-th equality restriction is infeasible, 0 != {}.", i, b_eq[i])));
                            }

                        }
                    }

                    // Check a_eq for fulll rank
                    let m = a_eq.nrows();
                    let r = QR::new(a_eq.clone()).r();
                    if r.row(m - 1).amax() < numerical_zero {
                        // QR factorization has a 0 row in R matrix with m <=n, therefore A_eq is rank deficient
                        return Err(crate::base::QPError::RankDeficient("Equality constraints are rank deficient, check for redundant constraints.".to_string()))
                    }

                    (m, a_eq, b_eq)

                }else{
                    return Err(crate::base::QPError::DimensionMismatch{matrix: "a_eq".to_string(), dims: (m2, n2), expected_dims:(m2, n)});
                }
            }else{
                return Err(QPError::DimensionMismatch{matrix: "b_eq".to_string(), dims: (m1, 1), expected_dims:(m2, 1)});
            }
        }else {
            if vec.len() > 0 {
                return Err(QPError::DimensionMismatch{matrix: "b_eq".to_string(), dims: (vec.len(), 1), expected_dims:(0, 1)});
            }
            (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
        }
    }else {
        if let Some(matrix) = &problem.a_eq {
            if matrix.nrows() > 0 {
                return Err(QPError::DimensionMismatch{matrix: "a_eq".to_string(), dims: matrix.shape(), expected_dims:(0, n)});
            }
        };
        (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
    };

    let (mut bounds, mut m_lb, mut m_ub) = if let Some(map) = &problem.bounds {
        // verify bounds & count bounds

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
                (Some(_), Some(_)) => {m_lb += 1; m_ub += 1;},
                (Some(_), None) => {m_lb += 1;},
                _ => {m_ub += 1;},
            }
        }

        (bounds, m_lb, m_ub)

    }else{
        (BTreeMap::new(), 0, 0)
    };

    let (m_ineq, a, b) = if let Some(vec) = &problem.b {
        let m1 = vec.len();
        if let Some(matrix) = &problem.a {
            let (m2, n2) = matrix.shape();
            if m1 == m2{
                if n == n2{
                    // Correct dims
                    let mut a = matrix.clone();
                    let mut b = vec.clone();

                    // Check a for 0 rows or singleton rows
                    for i in (0..m1).rev(){
                        let row = a.row(i);
                        let mut row_iter = row.iter().enumerate().filter(|(_, x)| {x.abs() < numerical_zero});
                        match row_iter.next() {
                            None => {
                                // 0 nonzero elements
                                if b[i] > -numerical_zero {
                                    a = a.remove_row(i);
                                    b = b.remove_row(i);
                                }else{
                                    return Err(crate::base::QPError::Infeasible(format!("{}-th inequality restriction is infeasible, 0 !<= {}.", i, b[i])));
                                }
                            },
                            Some((j, x)) => {
                                if let None = row_iter.next() {
                                    // Exactly 1 nonzero element. replace ineq with bound
                                    let bound = b[j] / x;

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
                                                    return Err(crate::base::QPError::Infeasible(format!("{}-th inequality restriction contradicts bounds for {}-th variable", i, j)))
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
                                                    return Err(crate::base::QPError::Infeasible(format!("{}-th inequality restriction contradicts bounds for {}-th variable", i, j)))
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
                                                        return Err(crate::base::QPError::Infeasible(format!("{}-th inequality restriction contradicts bounds for {}-th variable", i, j)))
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
                                                        return Err(crate::base::QPError::Infeasible(format!("{}-th inequality restriction contradicts bounds for {}-th variable", i, j)))
                                                    }
                                                }
                                                // looser lower bound
                                            }
                                        },
                                        _ => {
                                            // Variable is not bound, add the bound
                                            if x.is_sign_positive() {
                                                bounds.insert(j, (None, Some(bound)));
                                                m_ub += 1;
                                            } else {
                                                bounds.insert(j, (Some(bound), None));
                                                m_lb += 1;
                                            }

                                        },

                                    }

                                    // Remove the inequality
                                    a = a.remove_row(i);
                                    b = b.remove_row(i);

                                }
                                // More than 1 nonzero element
                            },
                        }
                    }

                    let m_ineq = a.nrows();
                    a = a.insert_rows(m_ineq, m_lb + m_ub, 0.0);
                    b = b.insert_rows(m_ineq, m_lb + m_ub, 0.0);

                    for (i, (j, v)) in bounds.iter().filter(|(_, v)| {matches!(v, (Some(x), None) if x.is_finite())}).enumerate(){
                        a[(m_ineq + i, *j)] = 1.0;
                        b[m_ineq + i] = v.0.unwrap();
                    }

                    for (i, (j, v)) in bounds.iter().filter(|(_, v)| {matches!(v, (None, Some(x)) if x.is_finite())}).enumerate(){
                        a[(m_ineq + m_lb + i, *j)] = -1.0;
                        b[m_ineq + m_lb + i] = -v.1.unwrap();
                    }


                    (m_ineq, a, b)

                }else{
                    return Err(QPError::DimensionMismatch{matrix: "a".to_string(), dims: (m2, n2), expected_dims:(m2, n)});
                }
            }else{
                return Err(QPError::DimensionMismatch{matrix: "b".to_string(), dims: (m1, 1), expected_dims:(m2, 1)});
            }
        }else {
            if vec.len() > 0 {
                
                return Err(QPError::DimensionMismatch{matrix: "b".to_string(), dims: (vec.len(), 1), expected_dims:(0, 1)});
            }
            (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
        }
    }else {
        if let Some(matrix) = &problem.a {
            if matrix.nrows() > 0 {
                return Err(QPError::DimensionMismatch{matrix: "a".to_string(), dims: matrix.shape(), expected_dims:(0, n)});
            }
        };
        (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
    };


    Ok(RawQP {
        hess,
        c,
        a,
        b,
        a_eq,
        b_eq,
        dims: (n, m_ineq + m_lb + m_ub, m_eq, m_ineq, m_lb, m_ub),
        x0: problem.x0.clone(),
        options: problem.options.clone()
    })
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
        let file = File::open("qp/presolve/minimal_problem.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();
        let raw_problem = presolve(&problem).unwrap();

        assert_eq!(raw_problem.dims, (3, 0, 0, 0, 0, 0));
        assert_eq!(raw_problem.a.shape(), (0, 3));
        assert_eq!(raw_problem.a_eq.shape(), (0, 3));
        assert_eq!(raw_problem.b.len(), 0);
        assert_eq!(raw_problem.b_eq.len(), 0);
    }

    #[test]
    fn presolve_rectangular_hess() {
        let file = File::open("qp/presolve/rectangular_hess.yaml").unwrap();
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
        let file = File::open("qp/presolve/rectangular_hess2.yaml").unwrap();
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
        let file = File::open("qp/presolve/dims_hess_c.yaml").unwrap();
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
}