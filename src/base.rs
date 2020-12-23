use nalgebra::{DMatrix, DVector, Cholesky};
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use thiserror::Error;
use util::*;
use std::fmt;

mod util;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConvexQP {
    pub hess: DMatrix<f64>,
    pub c: DVector<f64>,
    pub a: Option<DMatrix<f64>>,
    pub b: Option<DVector<f64>>,
    pub a_eq: Option<DMatrix<f64>>,
    pub b_eq: Option<DVector<f64>>,
    pub bounds: Option<BTreeMap<usize, (Option<f64>, Option<f64>)>>,
    pub x0: Option<DVector<f64>>,
    #[serde(default)]
    pub options: QPOptions,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QPOptions {
    pub algorithm: QPAlgorithm,
    pub max_iterations: usize,
    pub opt_tol: f64,
    pub step_tol: f64,
    pub fun_tol: f64,
    pub con_tol: f64,
    pub verbose: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QPSolution {
    pub x: DVector<f64>,
    pub eq_multipliers: Option<DVector<f64>>,
    pub ineq_multipliers: Option<DVector<f64>>,
    pub lb_multipliers: Option<DVector<f64>>,
    pub ub_multipliers: Option<DVector<f64>>,
    pub iterations: usize,
    pub fval: f64,
    pub first_order_cond: f64,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub enum QPAlgorithm {
    InteriorPoint,
    ActiveSet,
    //Other(fn(&ConvexQP) -> Result<QPSolution, QPError>),
}

impl fmt::Display for QPAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Serialize, Deserialize, Error, Debug, Clone)]
pub enum QPError {
    #[error("{algorithm} exceded maximum number of iterations. Last known solution had function value {fval} and first order condition {first_order_cond}")]
    MaxIterationsReached{
        algorithm: QPAlgorithm,
        last_soln: DVector<f64>,
        fval: f64,
        first_order_cond: f64,
    },
    #[error("Infeasibility detected: {0}")]
    Infeasible(String),
    #[error("Dimension error: {matrix} is {}x{} but should be {}x{}", dims.0, dims.1, expected_dims.0, expected_dims.1)]
    DimensionMismatch { 
        matrix: String, 
        dims: (usize, usize), 
        expected_dims: (usize, usize)
    },
    #[error("Rank deficiency found: {0}")]
    RankDeficient(String),
    #[error("{matrix} has NaN/∞ in entry ({row}, {col})")]
    Undefined{
        matrix: String,
        row: usize, 
        col: usize,
    },
}

impl Default for ConvexQP {
    fn default() -> ConvexQP {
        ConvexQP{
            hess: DMatrix::<f64>::identity(1,1),
            c: DVector::<f64>::zeros(1),
            a: None,
            b: None,
            a_eq: None,
            b_eq: None,
            bounds: None,
            x0: None,
            options: Default::default(),
        }
    }
}

impl Default for QPAlgorithm {
    fn default() -> QPAlgorithm { QPAlgorithm::InteriorPoint }
}

impl Default for QPOptions {
    fn default() -> QPOptions {
        QPOptions{
            algorithm: QPAlgorithm::InteriorPoint,
            max_iterations: 100,
            opt_tol: 1e-8,
            step_tol: 1e-8,
            fun_tol: 1e-8,
            con_tol: 1e-8,
            verbose: false,
        }
    }
}

impl ConvexQP {
    pub fn solve(&self) -> Result<QPSolution, QPError>{
        let raw = presolve(&self)?;
        
        if raw.dims.m == 0 {
            if raw.dims.m_eq == 0 {
                // No restrictions, attempt direct soln
                if let Some(chol) = Cholesky::new(raw.hess.clone()){
                    // Chol factorization worked, solve system
                    let x = chol.solve(&raw.c);
                    return Ok(QPSolution{
                        fval: 0.5*x.dot(&(&raw.hess*&x)) + x.dot(&raw.c),
                        x: x,
                        eq_multipliers: None,
                        ineq_multipliers: None,
                        lb_multipliers: None,
                        ub_multipliers: None,
                        iterations: 0,
                        first_order_cond: 0.0,
                    })
    
                }else{
                    // Chol factorization did not work, 0 eigenvalue, check for no solution or infinite solutions
                    return Err(QPError::Infeasible(format!("Unrestricted problem with strictly decreasing direction")))
                }
            }else{
                // Only eq restrictions, attempt soln via null space method
            }

        }

        match self.options.algorithm{
            QPAlgorithm::InteriorPoint => interior_point_method(raw),
            QPAlgorithm::ActiveSet => active_set_method(raw)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;
    use float_cmp::{ApproxEq, ApproxEqRatio};

    #[test]
    fn read_full_problem() {
        let file = File::open("qp/read/full_problem.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();

        assert_eq!(problem.hess[(0,0)], 1.0);
        assert_eq!(problem.hess[(1,0)], 0.0);
        assert_eq!(problem.hess[(1,1)], 2.0);
        assert_eq!(problem.hess[(2,2)], 3.0);

        assert_eq!(problem.c[0], 4.0);
        assert_eq!(problem.c[2], 6.0);

        assert!(matches!(problem.a, Some(_)));
        assert!(matches!(problem.b, Some(_)));
        assert!(matches!(problem.a_eq, Some(_)));
        assert!(matches!(problem.b_eq, Some(_)));
        assert!(matches!(problem.bounds, Some(_)));
        assert!(matches!(problem.x0, Some(_)));
        assert!(matches!(problem.options.algorithm, QPAlgorithm::InteriorPoint));

    }

    #[test]
    fn read_minimal_problem() {
        let file = File::open("qp/read/minimal_problem.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();

        assert!(matches!(problem.a, None));
        assert!(matches!(problem.b, None));
        assert!(matches!(problem.a_eq, None));
        assert!(matches!(problem.b_eq, None));
        assert!(matches!(problem.bounds, None));
        assert!(matches!(problem.x0, None));

        assert_eq!(problem.options.max_iterations, 100);
        assert!(!problem.options.verbose);
    }

    #[test]
    fn read_one_sided_bounds() {
        let file = File::open("qp/read/one_sided_bounds.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();

        let bounds = problem.bounds.unwrap();

        assert_eq!(bounds.get(&0).unwrap(), &(Some(0.0), Some(1.0)));
        assert_eq!(bounds.get(&1).unwrap(), &(Some(0.0), None));
        assert_eq!(bounds.get(&2).unwrap(), &(None, Some(0.0)));
    }

    #[test]
    fn read_yaml() {
        let file = File::open("qp/read/mid_problem.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();

        assert!(matches!(problem.a_eq, None));
        assert!(matches!(problem.b_eq, None));
        assert!(matches!(problem.x0, None));

        assert!(matches!(problem.a, Some(_)));
        assert!(matches!(problem.b, Some(_)));

        let bounds = problem.bounds.unwrap();

        assert_eq!(bounds.get(&0).unwrap(), &(Some(0.0), Some(1.0)));
        assert_eq!(bounds.get(&1).unwrap(), &(Some(f64::NEG_INFINITY), Some(f64::INFINITY)));
        assert_eq!(bounds.get(&2).unwrap(), &(Some(0.0), None));
    }

    #[test]
    fn maros_hs118() {
        let file = File::open("qp/maros/HS118.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();
        let mut problem_as = problem.clone();
        problem_as.options.algorithm = QPAlgorithm::ActiveSet;

        let soln = problem.solve().unwrap();
        let soln2 = problem_as.solve().unwrap();
        
        let fval = 664.82;
        println!("Optimum: {}\nInterior Point:\n\tfval: {}\n\titers: {}\nActive Set\n\tfval: {}\n\titers: {}", fval, soln.fval, soln.iterations, soln2.fval, soln2.iterations);
        assert!(fval.approx_eq_ratio(&soln.fval, 0.01));
        assert!(fval.approx_eq_ratio(&soln2.fval, 0.01));

        
        let compare_vecs = |x: &DVector<f64>, y: &DVector<f64>| {
            let n1 = x.amax();
            let n2 = y.amax();

            if !n1.is_normal() && n1.is_finite(){
                println!("{}, {}", n1, n2);
                n2 <=0.05
            }else if !n2.is_normal() && n2.is_finite() {
                println!("{}, {}", n1, n2);
                n1 <=0.05
            }else{
                
                let n = n1.max(n2);
                let diff_norm = (x - y).amax();
                println!("{}, {}, {}", n1, n2, diff_norm);
                (x - y).amax()/n <= 0.01
            }
        };
        assert!(compare_vecs(&soln.ineq_multipliers.unwrap(), &soln2.ineq_multipliers.unwrap()));
        assert!(compare_vecs(&soln.lb_multipliers.unwrap(), &soln2.lb_multipliers.unwrap()));
        assert!(compare_vecs(&soln.ub_multipliers.unwrap(), &soln2.ub_multipliers.unwrap()));
    }

    #[test]
    #[ignore = "Expensive, run in release"]
    fn maros_dualc1() {
        let file = File::open("qp/maros/DUALC1.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();
        let mut problem_as = problem.clone();
        problem_as.options.algorithm = QPAlgorithm::ActiveSet;

        let soln = problem.solve().unwrap();
        let soln2 = problem_as.solve().unwrap();
        
        let fval = 6.155250829462746e+03;
        println!("Optimum: {}\nInterior Point:\n\tfval: {}\n\titers: {}\nActive Set\n\tfval: {}\n\titers: {}", fval, soln.fval, soln.iterations, soln2.fval, soln2.iterations);
        assert!(fval.approx_eq_ratio(&soln.fval, 1e-2));
        assert!(fval.approx_eq_ratio(&soln2.fval, 1e-2));

        
        let compare_vecs = |x: &DVector<f64>, y: &DVector<f64>| {
            let n1 = x.amax();
            let n2 = y.amax();

            if !n1.is_normal() && n1.is_finite(){
                println!("{}, {}", n1, n2);
                n2 <=0.05
            }else if !n2.is_normal() && n2.is_finite() {
                println!("{}, {}", n1, n2);
                n1 <=0.05
            }else{
                
                let n = n1.max(n2);
                let diff_norm = (x - y).amax();
                println!("{}, {}, {}", n1, n2, diff_norm);
                (x - y).amax()/n <= 0.01
            }
        };
        assert!(compare_vecs(&soln.eq_multipliers.unwrap(), &soln2.eq_multipliers.unwrap()));
        assert!(compare_vecs(&soln.ineq_multipliers.unwrap(), &soln2.ineq_multipliers.unwrap()));
        assert!(compare_vecs(&soln.lb_multipliers.unwrap(), &soln2.lb_multipliers.unwrap()));
        assert!(compare_vecs(&soln.ub_multipliers.unwrap(), &soln2.ub_multipliers.unwrap()));
    }

    

    #[test]
    #[ignore = "Expensive, run in release"]
    fn maros_dualc2() {
        let file = File::open("qp/maros/DUALC2.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();
        let mut problem_as = problem.clone();
        problem_as.options.algorithm = QPAlgorithm::ActiveSet;

        let soln = problem.solve().unwrap();
        let soln2 = problem_as.solve().unwrap();
        
        let fval = 3.551307692684270e+03;

        
        
        println!("{}, {}, {}", fval, soln.fval, soln2.fval);
        assert!(fval.approx_eq_ratio(&soln.fval, 1e-2));
        assert!(fval.approx_eq_ratio(&soln2.fval, 1e-2));

        // epsilon if a number is zero, ratio otherwise
        let compare = |(x, y): (&f64, &f64)| {
            if x.is_normal() && y.is_normal() {
                x.approx_eq_ratio(y, 0.0001)
            }else{
                x.approx_eq(*y, (1e-4, 0))
            }
        };
        assert!(soln.eq_multipliers.unwrap().iter().zip(soln2.eq_multipliers.unwrap().iter()).all(compare));
        assert!(soln.ineq_multipliers.unwrap().iter().zip(soln2.ineq_multipliers.unwrap().iter()).all(compare));
        assert!(soln.lb_multipliers.unwrap().iter().zip(soln2.lb_multipliers.unwrap().iter()).all(compare));
        assert!(soln.ub_multipliers.unwrap().iter().zip(soln2.ub_multipliers.unwrap().iter()).all(compare));
        
    }
}