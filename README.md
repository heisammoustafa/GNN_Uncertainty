# Graph Neural Networks for Uncertainty Quantification in Micromagnetic Modeling

This repository contains research code for studying **uncertainty quantification (UQ)** in **graph neural networks (GNNs)** applied to **micromagnetic simulations**.

The goal of this project is to develop GNN-based surrogate models that predict magnetic properties, such as the **coercivity** of complex microstructures, while simultaneously providing **disentangled uncertainty estimates**, including both **aleatoric** and **epistemic** uncertainty.

By executing `main.py`, the code trains the models, evaluates predictive performance, computes uncertainty estimates, and generates corresponding **confidence curves** for assessing the quality of the uncertainty quantification.

## Paper

This repository accompanies the graph-neural-network study (Section 5 and Appendix 9) of the article [Modelling magnetic material properties with uncertainty-aware neural networks](https://doi.org/10.1016/j.commatsci.2026.115073) (*Computational Materials Science* 276, 115073, 2027). See [Citation](#citation) below.

## Installation

```bash
micromamba env create -f environment.yml
micromamba activate pub-gnn-unc
```

## Reproducing the results of the paper

The results and the figure of the graph-neural-network study (Fig. 18: parity/residual plots and confidence curve) are reproduced with

```bash
python reproduce_fig18.py
```

The script trains the model for ten seeds (123 to 132); a validation split is used once to select the number of epochs, which is then reused for all seeds. It uses a GPU if one is available. The output is written to `out_reproduce_fig18/`:

- `metrics_by_seed.csv`, `metrics_summary_across_seeds.csv`: MAE, MSE, RMSE and R² per seed and averaged
- `Hc_test_*_unc_hexbin.png` and `Hc_train_*_unc_hexbin.png`: parity and residual plots of the representative seed 123, coloured by the total, epistemic or aleatoric uncertainty (Fig. 18A)
- `confidence_curve_Hc_all_samples.png`: confidence curve over the ten seeds (Fig. 18B)
- `best_epoch.txt`, `weights_seed_*.pt`: number of epochs and the trained weights

Small deviations from the numbers in the paper can occur, depending on hardware and library versions.

`reproduce_fig18.py` and the plotting modules in `paper_plots/` are copies of `scripts/train_model_Hc_gnn_uncertainty.py` and `src/evaluation/` of the repository [wagerc97/magnetic-uncertainty](https://github.com/wagerc97/magnetic-uncertainty) (commit `9ed8c2d`, author: C. Wager), which uses this repository as a git submodule. Only the path handling and the imports were changed; the training and evaluation protocol is unchanged.

`python main.py` is a stand-alone example: it trains the model for ten seeds, writes the confidence curves to `confidence_plots/`, and then runs a 5-fold cross-validation with a parity plot (`cv_predictions`). It does not follow exactly the protocol of the paper. The plotting code of this repository (`model_evaluation.py`, `confidence.py`) produced the example figures in `confidence_plots/` and `Coercivity_Prediction_Uncertainty/`; these are not the figures of the paper.

Note on the other files in the root folder:

- `dataset_tr.pt`, `dataset_val.pt`, `dataset_te.pt` are caches written by `data_loader.py` (random 70/15/15 split). They are overwritten whenever `data_loader.py` is imported.
- `weights/best_model.pt` is an example checkpoint written by `train.py`.

## Data

`PKL/` contains 1054 synthetic Nd₂Fe₁₄B microstructures, generated with Neper and labelled with the coercivity μ₀H_c (in tesla) from a reduced-order micromagnetic model (see the references in [DATA_LICENSE.md](DATA_LICENSE.md)). Each microstructure is a graph: the nodes are grains (18 stored features per grain, e.g. grain size, aspect ratio, sphericity, crystallographic orientation and Stoner–Wohlfarth switching field), the edges connect neighbouring grains, and the edge feature is derived from the grain-boundary thickness. The feature construction is described in the paper and in the J. Magn. Magn. Mater. article cited below.

For each of the seven batches `N` (`1_V3`, `2_V3`, `3_V3`, `bigger`, `bigger2`, `edge_length_var`, `fisch_2`) there are three pickle files:

| File | Content |
|---|---|
| `data_x_N.pkl` | list of graphs; each graph is a list of grains with 18 features |
| `data_y_N.pkl` | list of labels, one `[μ₀H_c]` (in T) per graph |
| `edge_index_N.pkl` | list of graphs; each graph is a list of `[i, j]` grain index pairs |

Number of graphs per batch: 146 (`1_V3`), 17 (`2_V3`), 533 (`3_V3`), 26 (`bigger`), 79 (`bigger2`), 241 (`edge_length_var`), 12 (`fisch_2`).

The batches are not all stored on the same scale. In every batch except `fisch_2`, the node features 0, 1, 2, 3, 4 and 7 (counting from 0) are stored normalised by the box edge length (feature 14), to the powers 1, 1, 1, 3, 1 and 2, respectively, and `data_loader.py` undoes this normalisation when loading. `fisch_2` is stored without this normalisation.

The same files are archived on Zenodo together with the repository [Micromagnetic_GNN](https://github.com/heisammoustafa/Micromagnetic_GNN) ([10.5281/zenodo.23360688](https://doi.org/10.5281/zenodo.23360688)). `PKL.sha256` contains the checksums.

The data are included in the git repository, but not in the release archives (GitHub zip, Zenodo archive of this repository). If you downloaded such an archive, get the data with:

```bash
curl -L -o Micromagnetic_GNN-v1.0.1.zip "https://zenodo.org/api/records/23361847/files/heisammoustafa/Micromagnetic_GNN-v1.0.1.zip/content"
unzip -j Micromagnetic_GNN-v1.0.1.zip '*/PKL/*' -d PKL
sha256sum -c PKL.sha256
```

## Licensing

- **Code**: licensed under the [MIT License](LICENSE).
- **Data** (`PKL/`, `dataset_*.pt`, `weights/`, figures): licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [DATA_LICENSE.md](DATA_LICENSE.md) for details and the citation.

## Citation

If you use this work, please cite:

Wager, C., Moustafa, H., Kovacs, A., Ali, Q., Oezelt, H., Yamano, H., Yano, M., Sakuma, N., Hosoi, H., Kinoshita, A., Shoji, T., Kato, A., Schrefl, T. (2027). Modelling magnetic material properties with uncertainty-aware neural networks. *Computational Materials Science*, 276, 115073. [https://doi.org/10.1016/j.commatsci.2026.115073](https://doi.org/10.1016/j.commatsci.2026.115073)

If you use the data, please also cite the article in which the data were introduced:

Moustafa, H., Kovacs, A., Fischbacher, J., Gusenbauer, M., Ali, Q., Breth, L., Schrefl, T., Oezelt, H. (2025). Graph neural networks to predict coercivity and maximum energy product of hard magnetic microstructures. *Journal of Magnetism and Magnetic Materials*, 634, 173594. [https://doi.org/10.1016/j.jmmm.2025.173594](https://doi.org/10.1016/j.jmmm.2025.173594)

BibTeX entries are given in [DATA_LICENSE.md](DATA_LICENSE.md). Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

## Acknowledgement

This work was funded by the Austrian Science Fund (FWF) [10.55776/I6159](https://www.fwf.ac.at/forschungsradar/10.55776/I6159). The financial support by the Austrian Federal Ministry of Energy, Economy and Tourism, the National Foundation for Research, Technology and Development and the Christian Doppler Research Association is gratefully acknowledged.
