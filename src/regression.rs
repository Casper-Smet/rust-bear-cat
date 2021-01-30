/// Linear regression on one variable.
pub fn linear_regression(x_vec: &[f32], y: &[f32]) -> impl Fn(f32) -> f32 {
    assert_eq!(x_vec.len(), y.len());

    let n: f32 = x_vec.len() as f32;
    let xy_sum: f32 = x_vec
        .iter()
        .zip(y.iter())
        .fold(0.0, |acc, (x, y)| x * y + acc);

    let y_sum: f32 = y.iter().sum();
    let x_square: f32 = x_vec.iter().fold(0.0, |acc, x| acc + x.powf(2.0));
    let x_sum: f32 = x_vec.iter().sum();

    let a_num: f32 = y_sum * x_square - x_sum * xy_sum;
    let a_den: f32 = n * x_square - x_sum.powf(2.0);
    let a: f32 = a_num / a_den;

    let b_num: f32 = n * xy_sum - x_sum * y_sum;
    let b_den: f32 = n * x_square - x_sum.powf(2.0);
    let b: f32 = b_num / b_den;

    move |x: f32| a + b * x
}
