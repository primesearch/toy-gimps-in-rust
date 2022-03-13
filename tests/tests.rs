//! This tests toy-gimps-in-rust
//!
//! Josh McFerran, Winter 2022, Programming in Rust

use realfft::RealFftPlanner;
use rustfft::{num_complex::Complex, FftPlanner};

#[test]
fn test_fft() {
    let mut planner = FftPlanner::<f64>::new();
    for i in 0..23 {
        let signal_length = 1 << i;
        let fft = planner.plan_fft_forward(signal_length);
        let ifft = planner.plan_fft_inverse(signal_length);

        let mut example = Vec::new();
        for j in 0..signal_length {
            example.push(Complex {
                re: j as f64,
                im: 0.0,
            });
        }
        let start = example.clone();

        fft.process(&mut example);
        ifft.process(&mut example);
        example = example
            .iter()
            .map(|x| Complex {
                re: ((*x).re / example.len() as f64).round(),
                im: 0.0,
            })
            .collect();
        println!("Comparing fft with ifft at sig len 2 ^ {:?}", i);
        if !(start == example) {
            println!("Failed comparison:\n{:?}\nand\n{:?}", start, example);
            panic!()
        } else {
            println!("Good");
        }
    }
}

#[test]
fn test_real_fft() {
    let mut real_planner = RealFftPlanner::<f64>::new();
    for i in 0..23 {
        let signal_length = 1 << i;
        let fft = real_planner.plan_fft_forward(signal_length);
        let ifft = real_planner.plan_fft_inverse(signal_length);

        let mut example = Vec::new();
        for j in 0..signal_length {
            example.push(j as f64);
        }
        let start = example.clone();
        let mut fft_output = fft.make_output_vec();

        fft.process(&mut example, &mut fft_output).unwrap();
        ifft.process(&mut fft_output, &mut example).unwrap();
        example = example
            .iter()
            .map(|x| (x / example.len() as f64).round())
            .collect();
        println!("Comparing fft with ifft at sig len 2 ^ {:?}", i);
        if !(start == example) {
            println!("Failed comparison:\n{:?}\nand\n{:?}", start, example);
            panic!()
        } else {
            println!("Good");
        }
    }
}

#[test]
fn test_real_fft_problem_cases() {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let signal_length = 4;
    let fft = real_planner.plan_fft_forward(signal_length);
    let ifft = real_planner.plan_fft_inverse(signal_length);

    let mut example = vec![1.0, 5.0, 0.0, 0.0];
    let start = example.clone();
    let mut output = fft.make_output_vec();
    fft.process(&mut example, &mut output).unwrap();
    ifft.process(&mut output, &mut example).unwrap();
    example = example.iter().map(|x| (*x / signal_length as f64).round()).collect();
    
    println!("Non-squared case: Comparing\n{:?}\nwith\n{:?}", start, example);
    assert!(start == example);

    let squared: Vec<f64> = start.iter().map(|&x| x * x).collect();
    fft.process(&mut example, &mut output).unwrap();
    output = output.iter().map(|&x| (x * x) / signal_length as f64).collect();
    ifft.process(&mut output, &mut example).unwrap();
    example = example.iter().map(|x| (*x).round()).collect();

    println!("Squared case: Comparing\n{:?}\nwith\n{:?}", squared, example);
    assert!(squared == example);

}