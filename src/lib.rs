#[cfg(test)]
mod regression;

#[test]
fn reg() {
    let x = 30.0;
    let a = 65.14157;
    let b = 0.385225;
    let y = a + x * b;

    println!("{} + {} * {} = {}", a, b, x, y);

    let age: Vec<f32> = vec![43.0, 21.0, 25.0, 42.0, 57.0, 59.0];
    let glucose: Vec<f32> = vec![99.0, 65.0, 79.0, 75.0, 87.0, 81.0];
    let func = regression::linear_regression(&age, &glucose);
    let y_test = func(x);
    println!("{}", y_test);
    assert_eq!(y, y_test);
    
    let age: [f32; 6] = [43.0, 21.0, 25.0, 42.0, 57.0, 59.0];
    let glucose: [f32; 6] = [99.0, 65.0, 79.0, 75.0, 87.0, 81.0];
    let func = regression::linear_regression(&age, &glucose);
    let y_test = func(x);
    println!("{}", y_test);
    assert_eq!(y, y_test);
}
