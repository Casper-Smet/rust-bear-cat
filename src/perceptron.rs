#[derive(Debug)]
pub struct Node {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Node {
    pub fn activate(&self, input: &Vec<f32>) -> f32 {
        let total = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            + self.bias;
        if total >= 0. {
            1.
        } else {
            0.
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    pub nodes: Vec<Node>,
}

impl Layer {
    pub fn activate(&self, input: &Vec<f32>) -> Vec<f32> {
        self.nodes.iter().map(|n| n.activate(input)).collect()
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn activate(&self, input: Vec<f32>) -> Vec<f32> {
        let mut output = input;
        for l in self.layers.iter() {
            output = l.activate(&output);
        }
        output
    }
}
