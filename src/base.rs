use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use thiserror::Error;
use util::*;

mod util;

#[derive(Serialize, Deserialize, Debug)]
pub struct ConvexQP {
    hess: DMatrix<f64>,
    c: DVector<f64>,
    a: Option<DMatrix<f64>>,
    b: Option<DVector<f64>>,
    a_eq: Option<DMatrix<f64>>,
    b_eq: Option<DVector<f64>>,
    bounds: Option<BTreeMap<usize, (Option<f64>, Option<f64>)>>,
    x0: Option<DVector<f64>>,
    #[serde(default)]
    options: QPOptions,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QPOptions {
    algorithm: QPAlgorithm,
    max_iterations: usize,
    opt_tol: f64,
    step_tol: f64,
    fun_tol: f64,
    con_tol: f64,
    verbose: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QPSolution {
    x: DVector<f64>,
    eq_multipliers: Option<DVector<f64>>,
    ineq_multipliers: Option<DVector<f64>>,
    lb_multipliers: Option<DVector<f64>>,
    ub_multipliers: Option<DVector<f64>>,
    iterations: usize,
    fval: f64,
    first_order_cond: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum QPAlgorithm {
    InteriorPoint,
    ActiveSet,
    //Other(fn(&ConvexQP) -> Result<QPSolution, QPError>),
}

#[derive(Error, Debug)]
pub enum QPError {
    #[error("Maximum number of iterations reached. {msg}. Last known solution had function value {fval} and first order condition {first_order_cond}")]
    MaxIterationsReached{
        msg: String,
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
        interior_point_method(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;

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
        let soln = problem.solve().unwrap();
        
        let fval = 664.82;
        assert!(((soln.fval - fval)/fval).abs() < 0.01);
    }
}