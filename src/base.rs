use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;
use thiserror::Error;
use util::*;

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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum QPAlgorithm {
    InteriorPoint,
    ActiveSet,
    //Other(fn(&ConvexQP) -> Result<QPSolution, QPError>),
}

#[derive(Serialize, Deserialize, Error, Debug)]
pub enum QPError {
    #[error("{method} exceded maximum number of iterations. Last known solution had function value {fval} and first order condition {first_order_cond}")]
    MaxIterationsReached{
        method: String,
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
        assert!(((soln.fval - fval)/fval).abs() < 0.01);
        assert!(((soln2.fval - fval)/fval).abs() < 0.01);
    }

    #[test]
    #[ignore = "Expensive, run in release"]
    fn maros_dualc1() {
        let file = File::open("qp/maros/DUALC1.json").unwrap();
        let reader = BufReader::new(file);
        let mut problem: ConvexQP = serde_json::from_reader(reader).unwrap();
        problem.options.opt_tol = 1e-5;
        let mut problem_as = problem.clone();
        problem_as.options.algorithm = QPAlgorithm::ActiveSet;

        let soln = problem.solve().unwrap();
        let soln2 = problem_as.solve().unwrap();
        
        let fval = 6.155250829462746e+03;
        assert!(((soln.fval - fval)/fval).abs() < 0.01);
        assert!(((soln2.fval - fval)/fval).abs() < 0.01);
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
        assert!(((soln.fval - fval)/fval).abs() < 0.01);
        assert!(((soln2.fval - fval)/fval).abs() < 0.01);
    }
}