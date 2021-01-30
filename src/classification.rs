extern crate itertools;
use itertools::Itertools;

pub fn single_knn(k: usize, x: &[f32; 2], y: &[[f32; 3]]) -> Option<f32> {
    let mut distances: Vec<(f32, f32)> = y
        .iter()
        .map(|point: &[f32; 3]| (point[2], (x[0] - point[0]).hypot(x[1] - point[1])))
        .collect();
    distances.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

    let k_distances = distances[0..k].into_iter().map(|tuple| tuple.0 as u32);
    let unique_dist = k_distances.clone().unique();

    let unique_frequency = unique_dist.map(|unique_dist: u32| {
        (
            unique_dist,
            k_distances
                .clone()
                .filter(|dist: &u32| *dist == unique_dist)
                .count(),
        )
    });

    match unique_frequency.max_by_key(|(unique_dist, _)| *unique_dist) {
        Some(x) => return Some(x.0 as f32),
        None => return None,
    }
}

pub fn k_nearest_neighbours(k: usize, x: &[[f32; 2]], y: &[[f32; 3]]) -> Vec<Option<f32>> {
    x.iter()
        .map(|point: &[f32; 2]| single_knn(k, point, y))
        .collect()
}
