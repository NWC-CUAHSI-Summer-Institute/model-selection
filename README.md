# nextgen-model-selection
With many models to choose from for use in the Next Generation Water Resources Modeling Framework (NextGen), it's pertinent to have a data-based method for selecting which hydrologic model one will use in a given catchment. 

The nextgen-model-selection repository contains a [model selector](https://github.com/NWC-CUAHSI-Summer-Institute/nextgen-model-selection/blob/main/JAWRA_ModelSelector_Figure2.ipynb) that uses random forest to predict the performance of various hydrolgoic models in catchments using their catchment attributes, such as land cover, aridity, snow fraction, etc. 

The suggestions from the model grader are intended to act as a first pass at evaluating which model is suitable for a given catchment. 

Additionally, the repository includes code for exploring an ensemble modeling approach which combines the streamflow predictions of multiple hydrologic models. We compare this approach to that of any one hydrologic model involved in our study. 

This work was presented at AGU in 2022. The presentation is available [here](https://github.com/NWC-CUAHSI-Summer-Institute/nextgen-model-selection/blob/main/LBolotin_AGU_Poster_2022.pdf). 
