use criterion::{criterion_group, criterion_main, Criterion};
use quadprog_rs::base::{ConvexQP, QPOptions, QPAlgorithm};
use std::fs::{self, File};
use std::io::{self, BufReader};

pub fn interior_point_benchmark(c: &mut Criterion) -> io::Result<()> {
    let solver_options = QPOptions{
        algorithm: QPAlgorithm::InteriorPoint,
        max_iterations: 100,
        opt_tol: 1e-8,
        step_tol: 1e-8,
        fun_tol: 1e-8,
        con_tol: 1e-8,
        verbose: false,
    };
    let mut group = c.benchmark_group("int_point");
    
    for entry in fs::read_dir("qp\\maros")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && "json" == path.extension().unwrap(){
            let file = File::open(&path).unwrap();
            let reader = BufReader::new(file);
            let problem = { 
                let mut problem: ConvexQP = serde_json::from_reader(reader).unwrap();
                problem.options = solver_options;
                problem
            };
            group.bench_function(path.file_stem().unwrap().to_str().unwrap(), |b| b.iter(|| problem.solve()));
        }
    }

    group.finish();
    Ok(())
}

pub fn active_set_benchmark(c: &mut Criterion) -> io::Result<()> {
    let solver_options = QPOptions{
        algorithm: QPAlgorithm::ActiveSet,
        max_iterations: 100,
        opt_tol: 1e-8,
        step_tol: 1e-8,
        fun_tol: 1e-8,
        con_tol: 1e-8,
        verbose: false,
    };
    let mut group = c.benchmark_group("active_set");
    
    for entry in fs::read_dir("qp\\maros")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && "json" == path.extension().unwrap(){
            let file = File::open(&path).unwrap();
            let reader = BufReader::new(file);
            let problem = { 
                let mut problem: ConvexQP = serde_json::from_reader(reader).unwrap();
                problem.options = solver_options;
                problem
            };

            group.bench_function(path.file_stem().unwrap().to_str().unwrap(), |b| b.iter(|| problem.solve()));
        }
    }

    group.finish();
    Ok(())
}

pub fn active_set_fsp_benchmark(c: &mut Criterion) -> io::Result<()> {
    
    let mut group = c.benchmark_group("active_set_fsp");
    
    for entry in fs::read_dir("qp\\maros_feasible")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && "json" == path.extension().unwrap(){
            let file = File::open(&path).unwrap();
            let reader = BufReader::new(file);
            let problem: ConvexQP = serde_json::from_reader(reader).unwrap();
            group.bench_function(path.file_stem().unwrap().to_str().unwrap(), |b| b.iter(|| problem.solve()));
        }
    }
    
    group.finish();
    Ok(())
}


criterion_group!(benches, interior_point_benchmark);
criterion_main!(benches);