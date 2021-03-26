pub fn factorial(n: u128) -> u128 {
    match n {
        0 | 1 => 1,
        _ => factorial(n - 1) * n,
    }
}

#[allow(non_snake_case, unused_assignments, non_upper_case_globals)]
pub fn Q_rsqrt(number: f32) -> f32 {
    // Fast inverse square root: https://en.wikipedia.org/wiki/Fast_inverse_square_root
    let mut i: i64;
    let x2: f32;
    let mut y: f32;
    const threehalfs: f32 = 1.5;

    x2 = number * 0.5;
    y = number;
    i = number.to_bits() as i64;            // evil floating point bit level hacking
    i = 0x5f3759df - (i >> 1);              // what the fuck?
    y = f32::from_bits(i as u32);
    y = y * (threehalfs - (x2 * y * y));    // 1st iteration
    // y = y * (threehalfs - (x2 * y * y)); // 2nd iteration, this can be removed

    return y;
}
