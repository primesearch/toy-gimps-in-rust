//! Determines if the Mersenne number corresponding to the passed in exponent is
//! a prime number by running a PRP test with the IBDWT. See readme for details.
//!
//! Josh McFerran, Winter 2022, Programming in Rust

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// Retreives the command line args and checks them for validity before passing the bulk
/// of the work to prptest
fn main() {
    let (mut exponent, mut signal_length, mut update_frequency) = (0, 0, 0);
    let mut run_verbose = false;
    let mut args_list = std::env::args();
    args_list.next();
    while let Some(argument) = args_list.next() {
        match argument.as_str() {
            "-e" | "-E" | "-exp" | "--exponent" => {
                if let Some(next_arg) = args_list.next() {
                    exponent = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            "-s" | "-S" | "-sig" | "--siglen" => {
                if let Some(next_arg) = args_list.next() {
                    signal_length = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            "-f" | "-F" | "-freq" | "--frequency" => {
                if let Some(next_arg) = args_list.next() {
                    update_frequency = parsenum(next_arg);
                } else {
                    usage();
                }
            }
            "-v" | "-V" | "--verbose" => run_verbose = true,
            "-h" | "-H" | "--help" => usage_help(),
            &_ => usage(),
        }
    }

    assert!(exponent > 3, "Invalid exponent; must be greater than 3");
    assert!(
        signal_length > 3,
        "Invalid signal length; must be greater than 3"
    );
    assert!(
        signal_length < exponent,
        "Invalid signal length; must be less than the exponent"
    );

    if update_frequency == 0 {
        update_frequency = ((exponent as f64) / 100.0).ceil() as usize;
    }

    if prptest(exponent, signal_length, update_frequency, run_verbose) {
        println!("2 ^ {:?} - 1 is probably prime!", exponent);
    } else {
        println!("2 ^ {:?} - 1 is not prime.", exponent);
    }
}

/// This runs the actual loop and is the bulk of the program. Returns true if 2 ^ exponent - 1
/// is probably prime, false otherwise.
fn prptest(
    exponent: usize,
    signal_length: usize,
    update_frequency: usize,
    run_verbose: bool,
) -> bool {
    let (bit_array, two_to_the_bit_array) = init_bit_array(exponent, signal_length);
    let weight_array = init_weight_array(exponent, signal_length);
    let (fft, ifft) = init_fft(signal_length);
    let mut roundoff = 0.0;
    let mut max_roundoff = 0.0;

    let mut residue = signalize(3, &two_to_the_bit_array);

    let gec_l = 2000;
    let gec_l_2 = gec_l ^ 2;
    let mut gec_d = residue.clone();
    let mut gec_prev_d = residue.clone();
    let mut gec_saved_d = residue.clone();
    let mut gec_saved_residue = residue.clone();
    let mut gec_saved_i: usize = 0;
    let all_threes = vec![3.0; signal_length];

    if signal_length <= 5 {
        println!(
            "Bit Array: {:?}\ntwo_to_the_bit_array: {:?}\nWeight Array:{:?}\n",
            bit_array, two_to_the_bit_array, weight_array
        );
    } else {
        println!(
            "Bit Array: [{:?}, {:?}, {:?}, {:?}, {:?} ...]",
            bit_array[0], bit_array[1], bit_array[2], bit_array[3], bit_array[4]
        );
        println!(
            "two_to_the_bit_array: [{:?}, {:?}, {:?}, {:?}, {:?} ...]",
            two_to_the_bit_array[0],
            two_to_the_bit_array[1],
            two_to_the_bit_array[2],
            two_to_the_bit_array[3],
            two_to_the_bit_array[4]
        );
        println!(
            "Weight Array: [{:?}, {:?}, {:?}, {:?}, {:?} ...]\n",
            weight_array[0], weight_array[1], weight_array[2], weight_array[3], weight_array[4]
        );
    }

    let mut i = 0;
    while i < exponent {
        if i % update_frequency == 0 {
            println!("Iteration: {:?}", i);
            println!("{:.2}% Finished", (i as f64 / exponent as f64) * 100.0);
            println!("Current Roundoff Error: {:.4}", roundoff);
            if signal_length <= 5 {
                println!("Residue: {:?}\n", &residue);
            } else {
                println!(
                    "Residue: [{:?}, {:?}, {:?}, {:?}, {:?}, ...]\n",
                    residue[0], residue[1], residue[2], residue[3], residue[4]
                );
            }
        }
        if i % gec_l == 0 && i != 0 {
            gec_prev_d = gec_d.clone();
            let (temp_gec_d, temp_roundoff) = multmod_with_ibdwt(
                residue.clone(),
                gec_d,
                &two_to_the_bit_array,
                &weight_array,
                &fft,
                &ifft,
                run_verbose,
            );
            gec_d = temp_gec_d;
            if temp_roundoff > 0.4375 {
                eprintln!("Roundoff error is too great: {:.4} at iteration {:?} in the GEC L step. Try a higher signal length.", temp_roundoff, i);
                std::process::exit(1);
            }
        }
        if i % gec_l_2 == 0 && i != 0 {
            let mut check_value = gec_prev_d.clone();
            for j in 0..gec_l {
                let (temp_check_value, temp_roundoff) = squaremod_with_ibdwt(
                    check_value,
                    &two_to_the_bit_array,
                    &weight_array,
                    &fft,
                    &ifft,
                    run_verbose,
                );
                if temp_roundoff > 0.4375 {
                    eprintln!("Roundoff error is too great: {:.4} at iteration {:?} in the GEC L ^ 2 step {:?}. Try a higher signal length.", temp_roundoff, i, j);
                    std::process::exit(1);
                }
                check_value = temp_check_value;
            }
            let (temp_check_value, temp_roundoff) = multmod_with_ibdwt(
                check_value,
                all_threes.clone(),
                &two_to_the_bit_array,
                &weight_array,
                &fft,
                &ifft,
                run_verbose,
            );
            if temp_roundoff > 0.4375 {
                eprintln!("Roundoff error is too great: {:.4} at iteration {:?} in the GEC L ^ 2 multmod step. Try a higher signal length.", temp_roundoff, i);
                std::process::exit(1);
            }
            if temp_check_value != gec_d {
                gec_d = gec_saved_d.clone();
                residue = gec_saved_residue.clone();
                i = gec_saved_i;
            } else {
                gec_saved_d = gec_d.clone();
                gec_saved_residue = residue.clone();
                gec_saved_i = i;
            }
        }
        let (temp_residue, temp_roundoff) = squaremod_with_ibdwt(
            residue,
            &two_to_the_bit_array,
            &weight_array,
            &fft,
            &ifft,
            run_verbose,
        );
        residue = temp_residue;
        roundoff = temp_roundoff;
        max_roundoff = roundoff.max(max_roundoff);
        if roundoff > 0.4375 {
            eprintln!(
                "Roundoff error is too great: {:.4} at iteration {:?}. Try a higher signal length.",
                roundoff, i
            );
            std::process::exit(1);
        }

        i += 1;
    }

    if signal_length <= 5 {
        println!("\nFinal residue: {:?}", &residue);
    } else {
        println!(
            "\nFinal residue: [{:?}, {:?}, {:?}, {:?}, {:?}, ...]",
            residue[0], residue[1], residue[2], residue[3], residue[4]
        );
    }
    println!("Max Roundoff Error: {:.4}", max_roundoff);

    let first_is_nine = residue[0] == 9.0;
    let rest_are_zero = residue[1..signal_length].iter().all(|&x| x == 0.0);

    first_is_nine && rest_are_zero
}

fn squaremod_with_ibdwt(
    signal: Vec<f64>,
    two_to_the_bit_array: &[f64],
    weight_array: &[f64],
    fft: &Arc<dyn RealToComplex<f64>>,
    ifft: &Arc<dyn ComplexToReal<f64>>,
    run_verbose: bool,
) -> (Vec<f64>, f64) {
    let balanced_signal = balance(signal, two_to_the_bit_array);
    if run_verbose {
        println!("Balanced: {:?}", balanced_signal);
    }

    let transformed_signal = weighted_transform(balanced_signal, weight_array, fft);
    if run_verbose {
        println!("fft: {:?}", transformed_signal);
    }

    let squared_transformed_signal = transformed_signal.into_iter().map(|x| x * x).collect();
    if run_verbose {
        println!("Squared fft: {:?}", squared_transformed_signal);
    }

    let squared_signal = inverse_weighted_transform(squared_transformed_signal, weight_array, ifft);
    if run_verbose {
        println!("ifft: {:?}", squared_signal);
    }

    let rounded_signal: Vec<f64> = squared_signal.iter().map(|&x| x.round()).collect();
    if run_verbose {
        println!("Rounded: {:?}", rounded_signal);
    }
    let roundoff = squared_signal
        .iter()
        .zip(rounded_signal.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max);

    let carried_signal = complete_carry(rounded_signal, two_to_the_bit_array);
    if run_verbose {
        println!("Carried: {:?}\n\n", carried_signal);
    }

    (carried_signal, roundoff)
}

fn multmod_with_ibdwt(
    first_signal: Vec<f64>,
    second_signal: Vec<f64>,
    two_to_the_bit_array: &[f64],
    weight_array: &[f64],
    fft: &Arc<dyn RealToComplex<f64>>,
    ifft: &Arc<dyn ComplexToReal<f64>>,
    run_verbose: bool,
) -> (Vec<f64>, f64) {
    let first_balanced_signal = balance(first_signal, two_to_the_bit_array);
    if run_verbose {
        println!("First Balanced: {:?}", first_balanced_signal);
    }

    let first_transformed_signal = weighted_transform(first_balanced_signal, weight_array, fft);
    if run_verbose {
        println!("fft: {:?}", first_transformed_signal);
    }

    let second_balanced_signal = balance(second_signal, two_to_the_bit_array);
    if run_verbose {
        println!("Second Balanced: {:?}", second_balanced_signal);
    }

    let second_transformed_signal = weighted_transform(second_balanced_signal, weight_array, fft);
    if run_verbose {
        println!("fft: {:?}", second_transformed_signal);
    }

    let multiplied_transformed_signal = first_transformed_signal
        .into_iter()
        .zip(second_transformed_signal.into_iter())
        .map(|(x, y)| x * y)
        .collect();
    if run_verbose {
        println!("Squared fft: {:?}", multiplied_transformed_signal);
    }

    let squared_signal =
        inverse_weighted_transform(multiplied_transformed_signal, weight_array, ifft);
    if run_verbose {
        println!("ifft: {:?}", squared_signal);
    }

    let rounded_signal: Vec<f64> = squared_signal.iter().map(|&x| x.round()).collect();
    if run_verbose {
        println!("Rounded: {:?}", rounded_signal);
    }
    let roundoff = squared_signal
        .iter()
        .zip(rounded_signal.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max);

    let carried_signal = complete_carry(rounded_signal, two_to_the_bit_array);
    if run_verbose {
        println!("Carried: {:?}\n\n", carried_signal);
    }

    (carried_signal, roundoff)
}

fn balance(mut signal: Vec<f64>, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut carry_val = 0.0;
    for i in 0..signal.len() {
        signal[i] += carry_val;
        if signal[i] > two_to_the_bit_array[i] / 2.0 {
            signal[i] -= two_to_the_bit_array[i];
            carry_val = 1.0;
        } else {
            carry_val = 0.0;
        }
    }
    signal[0] += carry_val;
    signal
}

fn weighted_transform(
    signal: Vec<f64>,
    weight_array: &[f64],
    fft: &Arc<dyn RealToComplex<f64>>,
) -> Vec<Complex<f64>> {
    let mut complex_signal = fft.make_output_vec();
    let mut weighted_signal = Vec::new();

    for i in 0..signal.len() {
        weighted_signal.push(signal[i] * weight_array[i]);
    }
    fft.process(&mut weighted_signal, &mut complex_signal)
        .unwrap();

    complex_signal
}

fn inverse_weighted_transform(
    mut complex_signal: Vec<Complex<f64>>,
    weight_array: &[f64],
    ifft: &Arc<dyn ComplexToReal<f64>>,
) -> Vec<f64> {
    let mut real_signal = ifft.make_output_vec();
    let signal_length = real_signal.len() as f64;
    ifft.process(&mut complex_signal, &mut real_signal).unwrap();
    for i in 0..real_signal.len() {
        real_signal[i] /= weight_array[i] * signal_length;
    }
    real_signal
}

fn complete_carry(mut signal: Vec<f64>, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut carry_val = 0.0;
    let signal_length = signal.len();
    for i in 0..signal_length {
        signal[i] += carry_val;
        carry_val = (signal[i] / two_to_the_bit_array[i]).floor();
        signal[i] = ((signal[i] % two_to_the_bit_array[i]) + two_to_the_bit_array[i])
            % two_to_the_bit_array[i];
    }
    let mut i = 0;
    let mut index;
    while carry_val != 0.0 {
        index = i % signal_length;
        signal[index] += carry_val;
        carry_val = (signal[index] / two_to_the_bit_array[index]).floor();
        signal[index] = ((signal[index] % two_to_the_bit_array[index])
            + two_to_the_bit_array[index])
            % two_to_the_bit_array[index];
        i += 1;
    }

    signal
}

fn init_bit_array(exponent: usize, signal_length: usize) -> (Vec<f64>, Vec<f64>) {
    let fexponent: f64 = exponent as f64;
    let fsignal_length: f64 = signal_length as f64;
    let mut bit_array = Vec::new();

    let mut fi = 1.0;
    while fi < fsignal_length + 1.0 {
        bit_array.push(
            ((fexponent * fi) / fsignal_length).ceil()
                - ((fexponent * (fi - 1.0)) / fsignal_length).ceil(),
        );
        fi += 1.0;
    }

    assert!(
        bit_array[0] > 4.0,
        "Signal length too large for this exponent."
    );
    let mut two_to_the_bit_array = Vec::new();
    for &i in bit_array.iter() {
        two_to_the_bit_array.push(2_f64.powf(i));
    }
    (bit_array, two_to_the_bit_array)
}

fn init_weight_array(exponent: usize, signal_length: usize) -> Vec<f64> {
    let fexponent: f64 = exponent as f64;
    let fsignal_length: f64 = signal_length as f64;
    let mut weight_array = Vec::new();

    let mut fi = 0.0;
    while fi < fsignal_length {
        weight_array.push(2_f64.powf(
            ((fexponent * fi) / fsignal_length).ceil() - ((fexponent * fi) / fsignal_length),
        ));
        fi += 1.0;
    }
    weight_array
}

fn init_fft(signal_length: usize) -> (Arc<dyn RealToComplex<f64>>, Arc<dyn ComplexToReal<f64>>) {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal_length);
    let ifft = planner.plan_fft_inverse(signal_length);
    (fft, ifft)
}

fn signalize(mut num_to_signalize: usize, two_to_the_bit_array: &[f64]) -> Vec<f64> {
    let mut signal = vec![0.0_f64; two_to_the_bit_array.len()];
    let mut i = 0;
    while num_to_signalize > 0 {
        signal[i] = num_to_signalize as f64 % two_to_the_bit_array[i];
        num_to_signalize /= two_to_the_bit_array[i] as usize;
        i += 1;
    }
    signal
}

// fn designalize(signal: &[f64], two_to_the_bit_array: &[f64]) -> i64 {
//     let mut base = 1;
//     let mut resultant_num = 0;
//     for i in 0..signal.len() {
//         resultant_num += signal[i] as i64 * base;
//         base *= two_to_the_bit_array[i] as i64;
//     }
//     resultant_num
// }

/// Parse the given string as a `u128`.
fn parsenum(s: String) -> usize {
    s.parse().unwrap_or_else(|_| usage())
}

/// Print a usage error message and exit.
fn usage() -> ! {
    eprintln!(
        "Usage: toy-gimps-in-rust -e <exponent> -s <signal_length> [-f <update_frequency>] [-v]"
    );
    std::process::exit(1);
}

/// Print a help message and exit.
fn usage_help() -> ! {
    println!("Usage: toy-gimps-in-rust -e <exponent> -s <signal_length> [-f <update_frequency>] [-v] [-h]");
    println!("Checks if 2 ^ <exponent> - 1 is probably prime.");
    println!("  -e, --exponent <exponent>\t\tThe exponent to be checked");
    println!("  -s, --siglen <signal_length>\t\tThe signal length to use for the FFT");
    println!("  -f, --frequency <update_frequency>\tThe number of iterations between updates; defaults to <exponent> / 100");
    println!("  -v, --verbose\t\t\t\tRun with increased verbosity");
    println!("  -h, --help\t\t\t\tPrint this message and exit");
    std::process::exit(0);
}
