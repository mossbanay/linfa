/*
use linfa_ensemble::random_forest::hyperparameters::MaxFeatures;
use linfa_ensemble::RandomForest;
use linfa_predictor::{Predictor, ProbabilisticPredictor};
use linfa_trees::DecisionTreeParams;
use ndarray::{Array, Array1};
*/

fn main() {
    /*
    // Load data
    let data = vec![
        0.54439407, 0.26408166, 0.97446289, 0.81338034, 0.08248497, 0.30045893, 0.35535142,
        0.26975284, 0.46910295, 0.72357513, 0.77458868, 0.09104661, 0.17291617, 0.50215056,
        0.26381918, 0.06778572, 0.92139866, 0.30618514, 0.36123106, 0.90650849, 0.88988489,
        0.44992222, 0.95507872, 0.52735043, 0.42282919, 0.98382015, 0.68076762, 0.4890352,
        0.88607302, 0.24732972, 0.98936691, 0.73508201, 0.16745694, 0.25099697, 0.32681078,
        0.37070237, 0.87316842, 0.85858922, 0.55702507, 0.06624119, 0.3272859, 0.46670468,
        0.87466706, 0.51465624, 0.69996642, 0.04334688, 0.6785262, 0.80599445, 0.6690343,
        0.29780375,
    ];

    let xtrain = Array::from(data).into_shape((10, 5)).unwrap();
    let ytrain = Array1::from(vec![0, 1, 0, 1, 1, 0, 1, 0, 1, 1]);

    // Define parameters of single tree
    let tree_params = DecisionTreeParams::new(2)
        .max_depth(Some(3))
        .min_samples_leaf(2 as u64)
        .build();

    // Define parameters of random forest
    let ntrees = 300;
    let rf_params = RandomForestParamsBuilder::new(tree_params, ntrees)
        .max_features(Some(MaxFeatures::Auto))
        .build();
    let rf = RandomForest::fit(rf_params, &xtrain, &ytrain, None);

    let imp = rf.feature_importances();
    println!("Feature importances: {:?}", &imp);

    let most_imp_feat = imp.iter().max_by(|a, b| a.1.cmp(&b.1)).map(|(k, _v)| k);
    println!("Most important feature is ids={:?}", most_imp_feat);

    let preds = rf.predict(&xtrain);
    println!("Predictions: {:?}", &preds);

    let pred_probas = rf.predict_probabilities(&xtrain);
    println!("Prediction probabilities: {:?}", pred_probas);
    */
}
