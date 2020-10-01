use nalgebra::{DMatrix, DVector, QR};
use std::collections::HashMap;

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
    x0: DVector<f64>,
    options: crate::base::QPOptions,
}

pub(crate) fn presolve(problem: &crate::base::ConvexQP) -> Result<RawQP, crate::base::QPError>{
    
    // Check dimensions
    let (n, hess, c) = {
        let n1 = problem.c.len();
        let (n2, n3) = problem.hess.shape();
        if n2 == n3 {
            if n1 == n2{
                // TODO: Check hess convexity (?)
                (n1, problem.hess.symmetric_part(), problem.c.clone())
            }else{
                return Err(crate::base::QPError::DimensionMismatch(format!("hess is {}x{}, but c is {}x1", n2, n3, n1)));
            }
        }else{
            return Err(crate::base::QPError::DimensionMismatch(format!("hess should be square, but instead is {}x{}", n2, n3)));
        }
    };

    let (m_eq, a_eq, b_eq) = if let Some(vec) = &problem.b_eq {
        let m1 = vec.len();
        if let Some(matrix) = &problem.a_eq {
            let (m2, n2) = matrix.shape();
            if m1 == m2{
                if n == n2{
                    // Correct dims
                    let mut a_eq = matrix.clone();
                    let mut b_eq = vec.clone();
                    
                    // Check and remove zero rows
                    for i in (0..m1).rev(){
                        if a_eq.row(i).amax() < 1e-8 {
                            if b_eq[i].abs() < 1e-8 {
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
                    if r.row(m - 1).amax() < 1e-8 {
                        // QR factorization has a 0 row in R matrix with m <=n, therefore A_eq is rank deficient
                        return Err(crate::base::QPError::RankDeficient("Equality constraints are rank deficient, check for redundant constraints.".to_string()))
                    }

                    (m, a_eq, b_eq)

                }else{
                    return Err(crate::base::QPError::DimensionMismatch(format!("a_eq should be {0}x{1}, but instead is {0}x{2}.", m2, n, n2)));
                }
            }else{
                return Err(crate::base::QPError::DimensionMismatch(format!("b_eq has {0} rows, but a_eq has {1}.", m1, m2)));
            }
        }else {
            if vec.len() > 0 {
                return Err(crate::base::QPError::DimensionMismatch("Have b_eq, expected a_eq.".to_string()));
            }
            (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
        }
    }else {
        if let Some(matrix) = &problem.a_eq {
            if matrix.nrows() > 0 {
                return Err(crate::base::QPError::DimensionMismatch("Have a_eq, expected b_eq.".to_string()));
            }
        };
        (0, DMatrix::<f64>::zeros(0, n), DVector::<f64>::zeros(0))
    };

    let (mut bounds, mut m_lb, mut m_ub) = if let Some(map) = &problem.bounds {
        // verify bounds & count bounds
        
        let mut m_lb = 0;
        let mut m_ub = 0;

        let bounds: HashMap<_, _> = map.iter().filter_map(|(&k, &v)| { 
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
                (Some(x), _) if x.is_infinite() & x.is_sign_positive() => return Err(crate::base::QPError::Infeasible(format!("Variable {} has +∞ lower bound.", k))),
                (_, Some(x)) if x.is_infinite() & x.is_sign_negative() => return Err(crate::base::QPError::Infeasible(format!("Variable {} has -∞ upper bound.", k))),
                (Some(lb), Some(ub)) if lb > ub => return Err(crate::base::QPError::Infeasible(format!("Variable {} has inconsistent bounds [{}, {}].", k, lb, ub))),
                (Some(_), Some(_)) => {m_lb += 1; m_ub += 1;},
                (Some(_), None) => {m_lb += 1;},
                _ => {m_ub += 1;},
            }
        }
        
        (bounds, m_lb, m_ub)

    }else{
        (HashMap::new(), 0, 0)
    };

    

    

    

    unimplemented!()
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
}