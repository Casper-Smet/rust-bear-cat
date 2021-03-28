// 1. Choose the number of clusters(K) and obtain the data points
// 2. Place the centroids c_1, c_2, ..... c_k randomly
// 3. Repeat steps 4 and 5 until convergence or until the end of a fixed number of iterations
//       4. for each data point x_i:
//            - find the nearest centroid(c_1, c_2 .. c_k)
//            - assign the point to that cluster
//       5. for each cluster j = 1..k
//            - new centroid = mean of all points assigned to that cluster
// 6. End
// use ndarray::{array, Array, Dim, Axis};
use rand::Rng;

pub struct KMeans {
    n_clusters: u8,
    cluster_centers: Vec<Vec<f32>>,
}

impl KMeans {
    pub fn new(n_clusters: u8) -> KMeans {
        KMeans {
            n_clusters: n_clusters,
            cluster_centers: vec![],
        }
    }

    fn random_cluster_centers(&self, n_clusters: u8, n_features: &usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();

        (0..n_clusters)
            .map(|_: u8| {
                (0..*n_features)
                    .map(|_: usize| rng.gen::<f32>())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>()
    }
    fn distance_to_centroid(input: &Vec<f32>, cluster: &Vec<f32>) -> f32 {
        input.iter().zip(cluster).map(|(a, b)| (b - a).powf(2.)).sum::<f32>().sqrt()
    }

    fn best_centroid(&self, input:  &Vec<f32>) -> usize {
        let mut best_i: usize = 0;
        let mut best_dist: f32 = 0.;

        for i_val in self.cluster_centers.iter().enumerate() {
            let (i, cluster) = i_val;
            let dist = KMeans::distance_to_centroid(&input, cluster);
            if dist > best_dist {
                best_dist = dist;
                best_i = i;
            }
        }
        best_i
    }

    pub fn fit(&mut self, x: &Vec<Vec<f32>>, epochs: u128) {
        let n_features = x[0].len();
        self.cluster_centers = self.random_cluster_centers(self.n_clusters, &n_features);

        for _ in 0..epochs {
            let best_clusters = x.iter().map(|x_i| self.best_centroid(&x_i));
            
            let mut clusters: Vec<Vec<Vec<f32>>> = (0..self.n_clusters).map(|_| vec![vec![]]).collect();

            for row in best_clusters.zip(x) {
                let (cluster, x_i) = row;
                clusters[cluster].push(x_i.to_vec())
            }
            
            for i_val in clusters.iter().enumerate() {
                let (i, cluster) = i_val;
                let transpose: Vec<Vec<f32>> = (0..cluster[0].len()).map(|i| cluster.iter().map(|inner| inner[i].clone()).collect::<Vec<f32>>()).collect();
                self.cluster_centers[i] = transpose.iter().map(|coord| coord.iter().sum::<f32>() / n_features as f32).collect();
            }
        }
    }

    pub fn predict(&self, x: Vec<Vec<f32>>) -> Vec<usize> {
        x.iter().map(|x_i| self.best_centroid(&x_i)).collect()
    }
}
