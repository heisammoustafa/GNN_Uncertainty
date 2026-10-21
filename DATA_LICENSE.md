# Data License

The data in this repository are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Scope

This license applies to:

- `PKL/` (graphs, node features, edge lists and coercivity labels of 1054 synthetic Nd₂Fe₁₄B microstructures)
- `dataset_tr.pt`, `dataset_val.pt`, `dataset_te.pt` (PyTorch Geometric caches derived from `PKL/`)
- `weights/best_model.pt` (example model checkpoint)
- the figures in `confidence_plots/` and `Coercivity_Prediction_Uncertainty/`

The source code in this repository is **not** covered by this license. It is licensed separately under the MIT License (see [LICENSE](LICENSE)).

The published article (Wager et al., 2027) is open access under CC BY 4.0 (© 2026 The Authors). It is archived on Zenodo together with the exact figures of the paper (Fig. 18); see the README for the DOI.

## Origin of the data

The files in `PKL/` are identical (same SHA-256 checksums, see `PKL.sha256`) to the files in `PKL/` of the repository [Micromagnetic_GNN](https://github.com/heisammoustafa/Micromagnetic_GNN), which is archived on Zenodo: [10.5281/zenodo.23360688](https://doi.org/10.5281/zenodo.23360688). The microstructures were generated with Neper and labelled with the coercivity computed by a reduced-order micromagnetic model (Moustafa et al., AIP Advances 14, 025001, 2024; Moustafa et al., J. Magn. Magn. Mater. 634, 173594, 2025).

## You are free to

- **Share**: copy and redistribute the data in any medium or format.
- **Adapt**: remix, transform, and build upon the data for any purpose, including commercial use.

## Under the following terms

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **No additional restrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## Citation

If you use this data, please cite both articles:

> C. Wager, H. Moustafa, A. Kovacs, Q. Ali, H. Oezelt, H. Yamano, M. Yano, N. Sakuma, H. Hosoi, A. Kinoshita, T. Shoji, A. Kato, T. Schrefl,
> "Modelling magnetic material properties with uncertainty-aware neural networks",
> *Computational Materials Science*, vol. 276, p. 115073, 2027.
> DOI: [10.1016/j.commatsci.2026.115073](https://doi.org/10.1016/j.commatsci.2026.115073)

> H. Moustafa, A. Kovacs, J. Fischbacher, M. Gusenbauer, Q. Ali, L. Breth, T. Schrefl, H. Oezelt,
> "Graph neural networks to predict coercivity and maximum energy product of hard magnetic microstructures",
> *Journal of Magnetism and Magnetic Materials*, vol. 634, p. 173594, Nov. 2025.
> DOI: [10.1016/j.jmmm.2025.173594](https://doi.org/10.1016/j.jmmm.2025.173594)

```bibtex
@article{wager_modelling_2027,
	title = {Modelling magnetic material properties with uncertainty-aware neural networks},
	volume = {276},
	issn = {0927-0256},
	doi = {10.1016/j.commatsci.2026.115073},
	journal = {Computational Materials Science},
	author = {Wager, C. and Moustafa, H. and Kovacs, A. and Ali, Q. and Oezelt, H. and Yamano, H. and Yano, M. and Sakuma, N. and Hosoi, H. and Kinoshita, A. and Shoji, T. and Kato, A. and Schrefl, T.},
	year = {2027},
	pages = {115073},
}

@article{moustafa_graph_2025,
	title = {Graph neural networks to predict coercivity and maximum energy product of hard magnetic microstructures},
	volume = {634},
	issn = {03048853},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S0304885325008261},
	doi = {10.1016/j.jmmm.2025.173594},
	language = {en},
	journal = {Journal of Magnetism and Magnetic Materials},
	author = {Moustafa, H. and Kovacs, A. and Fischbacher, J. and Gusenbauer, M. and Ali, Q. and Breth, L. and Schrefl, T. and Oezelt, H.},
	month = nov,
	year = {2025},
	pages = {173594},
}
```

## Full license text

https://creativecommons.org/licenses/by/4.0/legalcode
