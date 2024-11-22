# nextgen-model-selection
With many models to choose from for use in the Next Generation Water Resources Modeling Framework (NextGen), it's pertinent to have a data-based method for selecting which hydrologic model one will use in a given catchment. 

The `nextgen-model-selection` repository contains a [model selector](https://github.com/NWC-CUAHSI-Summer-Institute/nextgen-model-selection/blob/main/JAWRA_ModelSelector_Figure2.ipynb) that uses random forest to predict the performance of various hydrologic models in catchments using their catchment attributes, such as land cover, aridity, snow fraction, etc. 

The suggestions from the model selector are intended to act as a first pass at evaluating which model is suitable for a given catchment, which is particularly helpful for large-sample modeling efforts. 

Additionally, the repository includes code for exploring an ensemble modeling approach which combines the streamflow predictions of multiple hydrologic models. We compare this approach to that of any one hydrologic model involved in our study, and find that ensemble modeling provides valuable predictive skill. 

This work was conducted as part of the CUAHSI National Water Center Summer Insitute. The full report is available [here](https://www.cuahsi.org/uploads/library/doc/SI2022_Report_v1.2.docx.pdf).
This work was presented at AGU in 2022. The presentation is available [here](https://github.com/NWC-CUAHSI-Summer-Institute/nextgen-model-selection/blob/main/LBolotin_AGU_Poster_2022.pdf). 

<img width="791" alt="Screenshot 2024-11-22 at 10 12 20 AM" src="https://github.com/user-attachments/assets/574eb0ec-e6c8-4acb-ac1c-0732408942ce">
