---
title: '*pyDeltaRCM*: a flexible numerical delta model'
tags:
  - Python
  - sedimentology
  - deltas
  - stratigraphy
authors:
  - name: Andrew J. Moodie
    orcid: 0000-0002-6745-036X
    affiliation: "1"
  - name: Jayaram Hariharan
    orcid: 0000-0002-1343-193X
    affiliation: "1"
  - name: Eric Barefoot
    orcid: 0000-0001-5770-2116
    affiliation: "2"
  - name: Paola Passalacqua
    affiliation: "1"
    orcid: 0000-0002-4763-7231
affiliations:
  - name: Department of Civil, Architectural, and Environmental Engineering, University of Texas at Austin, Austin, TX, USA
    index: 1
  - name: Department of Earth, Environmental and Planetary Sciences, Rice University, Houston, TX, USA
    index: 2
date: 08 June 2021
bibliography: paper.bib
---

# Summary

River deltas provide many societal benefits, and sustainability of these landforms may be impacted by human modification and global climate change.
Reduced-complexity numerical delta models incorporate limited physical processes, allowing researchers to assess the spatiotemporal evolution of landscape response to individual processes and environmental forcings. 
This is useful to understand, for example, shifting delta morphology due to sea-level rise, changing vegetal cover, or flooding intensity.
As a result, many numerical delta models have been proposed in the literature, and results from these studies are difficult to compare because of various design and implementation choices.
*pyDeltaRCM* (`v2.x`) delivers a computationally efficient and easy-to-customize implementation of the DeltaRCM numerical model [@liang_reduced_1_2015], enabling comparison and reproducibility in studies of delta change due to various environmental forcings.


# Statement of need

River deltas are societally important landforms because they provide arable land, deep inland ports, and are home to hundreds of millions of people globally [@edmonds_coastal_2020].
Existing at the interface between landmasses and water bodies, deltas are impacted by a multitude of processes arising in both of these domains.
For example, changes in sediment input to the delta modulate the rate at which new land is built; similarly, rising water levels in the downstream basin create flooded land.
In addition to natural processes, human landscape modification renders deltaic environments more sensitive to global climate change into the future [@paola_natural_2011].
Demand to understand natural delta processes, and how these processes will respond to various  environmental forcings, has led to a proliferation of numerical delta models in the literature [@overeem_three_2005].

The DeltaRCM delta model [@liang_reduced_1_2015] has gained popularity among geomorphologists due to an attractive balance of computational cost, realism, and interpretability [@larsen_appropriate_2016]. 
For example, studies have employed the DeltaRCM design to examine delta morphology and dynamism response to sea-level rise and regional subsidence [@liang_quantifying_2016; @liang_how_2016], as well as extended model design to simulate delta evolution with vegetation [@lauzon_comparing_2018] and ice and permafrost [@lauzon_ice_2019; @piliouras_unraveling_2021].
However, comparison among these studies is difficult, owing to disparate code bases, various implementation choices, lack of version control, and proprietary software dependencies.


# Background

Here, version 2.x of *pyDeltaRCM* is introduced; *pyDeltaRCM* is a computationally efficient, free and open source, and easy-to-customize numerical delta model based on the original DeltaRCM design.
The original DeltaRCM framework is inspired by well-understood physical phenomena, and models mass movement as a probabilistic weighted random-walk process coupled with a set of hierarchical rules; the model is extensively described in @liang_reduced_1_2015 and @liang_reduced_2_2015.

This same framework is the basis for *pyDeltaRCM* v2.x, with a few modifications selected only to resolve known numerical instabilities, improve computational efficiency, and support reproducible simulations.
*PyDeltaRCM* depends only on common Python packages `numpy` [@harris2020], `matplotlib` [@hunter2007], `scipy` [@virtanen2020], `netCDF4`, `pyyaml`, and `numba` [@lam_numba_2015].

![Simulation with *pyDeltaRCM* v2.x, default parameter set, and random `seed: 10151919`. Simulation was run for 4000 timesteps, and assumes 10 days of bankfull discharge per year. \label{fig:timeseries}](figures/timeseries.png)


# Flexible and easy to use

*pyDeltaRCM* is an object-oriented package, providing the central model class `DeltaModel`.
By creating custom model behavior as subclasses of `DeltaModel`, researchers can easily add, subtract, and modify model components without altering code that is not pertinent to the science objective.
Importantly, separating custom code from core model code makes clear how different studies can be compared.
The *pyDeltaRCM* documentation provides several examples for how to implement custom model behavior on top of the core `DeltaModel` object.

*pyDeltaRCM* also provides infrastructure to accelerate scientific exploration, such as the ability to configure multiple simulations from a single file.
A preprocessor orchestrates `parallel` simulations for multi-core systems (optionally), and implements several tools to support simulations exploring a parameter space.
For example, `matrix` expansion converts lists of parameters into an n-dimensional set of simulations.
Similarly, replicate simulations can be created via an `ensemble` specification.

Reproducibility and computational efficiency were important priorities in *pyDeltaRCM* development.
For example, to-disk logging records all parameters, system-level and version data, and random-seed information to ensure that all runs can be recreated.
Additionally, "checkpoint" infrastructure has been added to the model, which records simulation progress during computation and can later resume model runs for further simulation.
Finally, *pyDeltaRCM* uses `numba` for computational optimization [@lam_numba_2015], and does not depend on any proprietary software.

*pyDeltaRCM* component units and integrations are thoroughly documented and tested.
Component-level documentation describes implementation notes, whereas narratives in "Guide" and "Example" documentation describes high-level model design and best practices for model use and development.
*pyDeltaRCM* also couples with other numerical models via the CSDMS Basic Model Interface 2.0 [@hutton_basic_2020; @BMI_pyDeltaRCM].


# Acknowledgments

We gratefully acknowledge Rebecca Lauzon and Mariela Perignon for developing an implementation of *DeltaRCM* in Python that was the basis for *pyDeltaRCM*. 
We also thank the National Science Foundation for supporting us in developing this software, by way of a Postdoctoral Fellowship to A.M. (EAR 1952772) and a grant to J.H. and P.P. (EAR 1719670).


# References