pub fn e_approximation(n: u128) -> f64 {
    // Approximation of Eulers number
    (0..n).fold(0.0, |a, b| a + 1. / factorial(b) as f64)
}

fn factorial(n: u128) -> u128 {
    match n {
        0 | 1 => 1,
        _ => factorial(n - 1) * n,
    }
}
