use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize, Deserializer};
use std::collections::HashMap;

mod util;

#[derive(Serialize, Deserialize, Debug)]
struct ConvexQP {
    hess: DMatrix<f64>,
    c: DVector<f64>,
    a: Option<DMatrix<f64>>,
    b: Option<DVector<f64>>,
    a_eq: Option<DMatrix<f64>>,
    b_eq: Option<DVector<f64>>,
    bounds: Option<HashMap<usize, (Option<f64>, Option<f64>)>>,
    x0: Option<DVector<f64>>,
    #[serde(default)]
    options: QPOptions,
}

#[derive(Serialize, Deserialize, Debug)]
struct QPOptions {
    algorithm: QPAlgorithm,
    max_iterations: usize,
    opt_tol: f64,
    step_tol: f64,
    fun_tol: f64,
    con_tol: f64,
    verbose: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct QPSolution {
    x: DVector<f64>,
    eq_multipliers: Option<DVector<f64>>,
    ineq_multipliers: Option<DVector<f64>>,
    lb_multipliers: Option<DVector<f64>>,
    ub_multipliers: Option<DVector<f64>>,
    iterations: usize,
    fval: f64,
    first_order_cond: f64,
}

#[derive(Serialize, Deserialize, Debug)]
enum QPAlgorithm {
    InteriorPoint,
    ActiveSet,
    //Other(fn(&ConvexQP) -> Result<QPSolution, QPError>),
}

#[derive(Debug)]
enum QPError {
    MaxIterationsExceeded(String),
    Infeasible(String),
    DimensionMismatch(String),
    Other(String),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn read_full_problem() {
        let file = File::open("sample_problems/full_problem.json").unwrap();
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
        let file = File::open("sample_problems/minimal_problem.json").unwrap();
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
        let file = File::open("sample_problems/one_sided_bounds.json").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_json::from_reader(reader).unwrap();

        let bounds = problem.bounds.unwrap();

        assert_eq!(bounds.get(&0).unwrap(), &(Some(0.0), Some(1.0)));
        assert_eq!(bounds.get(&1).unwrap(), &(Some(0.0), None));
        assert_eq!(bounds.get(&2).unwrap(), &(None, Some(0.0)));
    }

    #[test]
    fn read_yaml() {
        let file = File::open("sample_problems/mid_problem.yaml").unwrap();
        let reader = BufReader::new(file);
        let problem: ConvexQP = serde_yaml::from_reader(reader).unwrap();

        assert!(matches!(problem.a_eq, None));
        assert!(matches!(problem.b_eq, None));
        assert!(matches!(problem.x0, None));

        assert!(matches!(problem.a, Some(_)));
        assert!(matches!(problem.b, Some(_)));

        let bounds = problem.bounds.unwrap();

        assert_eq!(bounds.get(&0).unwrap(), &(Some(0.0), Some(1.0)));
        assert_eq!(bounds.get(&1).unwrap(), &(Some(-f64::INFINITY), Some(0.0)));
        assert_eq!(bounds.get(&2).unwrap(), &(Some(0.0), None));
    }
}