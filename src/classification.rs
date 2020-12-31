extern crate itertools;
use itertools::Itertools;

pub fn single_knn(k: usize, x: &[f32;2], y: &[[f32;3]]) -> f32 {
    let mut distances: Vec<f32> = vec![Default::default(); y.len()];
    for (j, val) in y.iter().enumerate() {
        distances[j] = ((x[0] - val[0]).powf(2.0) + (x[1] - val[1]).powf(2.0)).sqrt();
    }
    let mut distances: Vec<(f32, f32)> = y.iter().map(|point: &[f32;3]| (point[2], 
        (
            (x[0] - point[0]).powf(2.0) + 
            (x[1] - point[1]).powf(2.0)
        ).sqrt())
    ).collect();
    distances.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

    let k_distances: Vec<u32> = distances[0..k].iter().map(|tuple| tuple.0 as u32).collect();
    let unique_dist: Vec<u32> = k_distances.iter().cloned().unique().collect_vec();

    let unique_frequency: Vec<(&u32, usize)> = unique_dist.iter().map(|unique_dist: &u32| (unique_dist, k_distances.iter().filter(|dist: &&u32| *dist == unique_dist).count())).collect_vec();
    
    *unique_frequency.iter().max_by_key(|(unique_dist, _)| unique_dist).expect("Nothing found").0 as f32
}

pub fn k_nearest_neighbours(k: usize, x: &[[f32;2]], y: &[[f32;3]]) -> Vec<f32> {
    return x.iter().map(|point: &[f32;2]| single_knn(k, point, y)).collect::<Vec<f32>>();
}