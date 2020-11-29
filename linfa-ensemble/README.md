# Ensemble learning

`linfa-ensemble` is a crate in the [`linfa`](https://crates.io/crates/linfa) ecosystem, an effort to create a toolkit for classical Machine Learning implemented in pure Rust, akin to Python's `scikit-learn`.

## Current state

`linfa-ensemble` currently provides an implementation of
* random forest fitting
* voting classifier

## Examples

There is an example in the `examples/` directory how to use random forest models. To run, use:

```bash
$ cargo run --release --example random_forest
```

The current benchmark performs random forest with 10, 100, 500, 1000 independent trees to predict labels and compare such results to ground truth.


## License
Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <http://opensource.org/licenses/MIT>, at your option. This file may not be copied, modified, or distributed except according to those terms.