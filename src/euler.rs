use crate::math::factorial;
use rand::Rng;

pub fn e_approximation(n: u128) -> f64 {
    // Approximation of Eulers number
    (0..n).fold(0.0, |a, b| a + 1. / factorial(b) as f64)
}

fn just_once() -> f64 {
    let mut counter: f64 = 0.;
    let mut summer: f64 = 0.;
    let mut rng = rand::thread_rng();
    while summer <= 1. {
        summer += rng.gen::<f64>();
        counter += 1.;
    }
    counter
}

pub fn random_e(n: u128) -> f64 {
    (0..n).map(|_| just_once()).sum::<f64>() / (n as f64)
}