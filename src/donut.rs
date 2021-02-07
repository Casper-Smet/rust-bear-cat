use ndarray::Array;
use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::OwnedRepr;

const SCREEN_SIZE: usize = 40;
const THETA_SPACING: f32 = 0.07;
const PHI_SPACING: f32 = 0.02;
const ILLUMINATI: [&str; 12] = [".", ",", "-", "~", ":", ";", "=", "!", "*", "#", "$", "@"];

const R1: f32 = 1.;
const R2: f32 = 2.;
const K2: f32 = 5.;
const K1: f32 = SCREEN_SIZE as f32 * K2 * 3. / (8. * (R1 + R2));

pub fn render_frame(a: f32, b: f32) -> ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>> {
    // Returns a frame of the spinning 3D donut.
    // Based on the pseudocode from: https://www.a1k0n.net/2011/07/20/donut-math.html

    let cos_a: f32 = a.cos();
    let sin_a: f32 = a.sin();
    let cos_b: f32 = b.cos();
    let sin_b: f32 = b.sin();

    let mut output = Array::from_elem((SCREEN_SIZE, SCREEN_SIZE), " ");
    let mut zbuffer = Array::<f32, _>::zeros((SCREEN_SIZE, SCREEN_SIZE));
    let phi = Array::range(0., 2. * std::f32::consts::PI, PHI_SPACING);
    let cosphi = phi.mapv(f32::cos);
    let sinphi = phi.mapv(f32::sin);

    let theta = Array::range(0., 2. * std::f32::consts::PI, THETA_SPACING);
    let costheta = theta.mapv(f32::cos);
    let sintheta = theta.mapv(f32::sin);

    let circle_x = R2 + R1 * &costheta;
    let circle_y = R1 * &sintheta; // (90,)

    let circle_y_ab_sin = &circle_y * cos_a * sin_b;
    let circle_y_ab_cos = &circle_y * cos_a * cos_b;
    let cos_plus_sin = cos_b * &cosphi + sin_a * sin_b * &sinphi;
    let sin_min_cos = sin_b * &cosphi - sin_a * cos_b * &sinphi;

    for i in 0..90 {
        let x = circle_x[i] * &cos_plus_sin - circle_y_ab_sin[i];
        let y = circle_x[i] * &sin_min_cos + circle_y_ab_cos[i];
        let z = K2 + cos_a * circle_x[i] * &sinphi + circle_y[i] * sin_a;
        let ooz = 1. / z; // One over z
        let xp: Array<f32, _> = SCREEN_SIZE as f32 / 2. + K1 * &ooz * x; //() -> mapv
        let xp = xp.mapv(|elem| elem as usize);
        let yp = SCREEN_SIZE as f32 / 2. - K1 * &ooz * y;
        let yp = yp.mapv(|elem| elem as usize);
        let luminance =
            (costheta[i] * &cosphi * sin_b - cos_a * &costheta[i] * &sinphi - sin_a * &sintheta[i]
                + cos_b * (cos_a * &sintheta[i] - &costheta[i] * sin_a * &sinphi))
                * 8.;
        let luminance = luminance.mapv(|elem| elem.round() as i32); // Round to closest integer, cast to integer

        for i in 0..315 {
            if (luminance[i] >= 0) & (ooz[i] > zbuffer[[xp[i], yp[i]]]) {
                zbuffer[[xp[i], yp[i]]] = ooz[i];
                output[[xp[i], yp[i]]] = ILLUMINATI[luminance[i] as usize];
            }
        }
    }
    output
}

pub fn run_donut(mut a: f32, mut b: f32) {
    // Renders multiple frames of 3D donut

    for _ in 0..SCREEN_SIZE * SCREEN_SIZE {
        a += THETA_SPACING;
        b += PHI_SPACING;
        let frame = render_frame(a, b);
        print!("\x1b[H");
        for row in frame.outer_iter() {
            println!("{}", row.to_vec().join(" "));
        }
    }
}
