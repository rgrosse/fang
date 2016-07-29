
This repository gives the code for reproducing the experiments in [Grosse and Salakhutdinov, 2015, "Scaling up natural gradient by sparsely factorizing the inverse Fisher matrix"](http://www.cs.toronto.edu/~rgrosse/icml2015-fang.pdf). The code is not optimized for low-level efficiency issues.

Setup:

1. Clone this repository
2. Install [CUDAMat](https://github.com/cudamat/cudamat) and [GNumpy](http://www.cs.toronto.edu/~tijmen/gnumpy.html)
3. Create `config.py` based on `config_example.py`, and fill in the required directories.

The following experiments use as an example training an RBM on MNIST using SGD. 

To train the RBM:

* from the command line:

        python experiments/from_scratch.py mnist_full/sgd

* Alternatively, in interactive mode, with visualizations and likelihood evaluations as training progresses:

        from experiments import from_scratch
        from_scratch.run('mnist_full/sgd', True)

To estimate the RBM partition function and save approximate samples (by running AIS followed by a lot of Gibbs steps):

    python experiments/evaluation.py from_scratch/mnist_full/sgd all

To plot the log-likelihoods as a function of time:

    from experiments import plotting
    plotting.show_comparison('mnist_full', ['sgd'], 'test')

To plot the approximate samples and save them to `<config.FIGURES_DIR>/evaluation/from_scratch/`:

    from experiments import evaluation
    evaluation.save_figures('from_scratch/mnist_full/sgd')

In the above commands, `mnist_full` can be replaced with `mnist_quick` for a faster experiment with worse performance, or with `omniglot_full` for the Omniglot dataset. (This dataset will be added the repository soon.) The training curves shown in the paper correspond to `mnist_full` and `omniglot_full`.

To generate the eigenspectrum visualization in Figure 2:

    from experiments import fisher_vis
    fisher_vis.generate_eigenvalue_figure()

To compare KL divergences as in Table 1:

    from experiments import fisher_vis
    fisher_vis.compare_kldiv()

Note: the `rand conn` results will differ slightly from the paper since I forgot to seed the RNG.

