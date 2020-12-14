use linfa::{
    error::{Error, Result},
    Float, Label,
};
use linfa_trees::DecisionTreeParams;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy)]
pub enum MaxFeatures {
    Sqrt,
    Log2,
    All,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy)]
pub struct RandomForestParams<F, L> {
    pub n_estimators: usize,
    pub tree_hyperparameters: DecisionTreeParams<F, L>,
    pub max_features: MaxFeatures,
    pub use_bootstrapping: bool,
}

impl<F: Float, L: Label> RandomForestParams<F, L> {
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    pub fn tree_hyperparameters(mut self, tree_hyperparameters: DecisionTreeParams<F, L>) -> Self {
        self.tree_hyperparameters = tree_hyperparameters;
        self
    }

    pub fn max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    pub fn use_bootstrapping(mut self, use_bootstrapping: bool) -> Self {
        self.use_bootstrapping = use_bootstrapping;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.n_estimators == 0 {
            return Err(Error::Parameters(
                "Number of estimators cannot be zero".to_string(),
            ));
        }

        self.tree_hyperparameters.validate()
    }
}
