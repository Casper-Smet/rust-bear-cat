use crate::math::factorial;

pub fn e_approximation(n: u128) -> f64 {
    // Approximation of Eulers number
    (0..n).fold(0.0, |a, b| a + 1. / factorial(b) as f64)
}
