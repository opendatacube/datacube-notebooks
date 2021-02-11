# Earth Observation Notebooks

## Jupyter notebooks
A Jupiter notebook is a research document containing live code, annotated write-ups and descriptions of analyses or research.    

A Jupyter notebook aims to make research re-producible and re-usable by offering a common programmatic interface whereby all diagrams, tables, plots are results of inline executed code. 

Being able to read the code that generates results adds full transparency to research as well as promotes the adoption and re-use of that research.  

## This repository

This repository houses various Jupyter notebooks from the open data cube community. A varied community of remote sensing, GIS, and earth observation specialists/researchers. 

The [open data cube](https://www.opendatacube.org/) is used in these notebooks to query large data-sets for time series rasters on which analysis is conducted.  
<br>

## List of notebooks 
<br>  

- **Loading Data**  
 
   This notebook details retrieval of data from the  [`open data cube`](https://www.opendatacube.org/).  Topics include establishing a connection to the data cube, defining what data gets loaded, and a high level description of the `xarray` object returned by the load operation   
  
  >Link: [datacube load tutorial](./Load%20Tutorial/loading_from_data_cube.ipynb)  
Data:  `GPM`   

- **Forest Degradation using Linear Regression Analysis**  

   This notebook runs regression on an NDVI time series. Slope of a regressed line is used as proxy to determine vegetation gain or loss. Based on the publication *Assessment of Forest Degradation in Vietnam Using Landsat Time Series Data* by Vogelmann Et al.  
    
  > Link: [forest degredation](./Forest%20Degredation%20using%20Regression%20Analysis%20on%20NDVI/Forest_Degradation_Vogelmann_et_al.ipynb)  
  > Data: `Landsat 7 Collection 1`


- **Cloud Analysis on Landsat**  

  This notebook compiles a series of summaries and visualizations regarding cloud coverage on landsat imagery. 
  > Link: [cloud analysis](./Cloud%20Analysis)  
  > Data: `Landsat 7 Collection 1`  

- **Coastal Change**
   This directory houses notebooks on coastal analysis, including a very simple coastline classifier.   
  > Link: [coastal_change](./Coastal%20Change)  
  > Data: `Landsat 7 Collection 1`  
   
- **Land Change Detection on ALOS imagery**
  This directory houses a change detection case-study in vietnam on ALOS imagery.   
  > Link: [land change](./Land%20Change%20Detection%20on%20ALOS%20imagery)  
  > Data: `ALOS2 PALSAR2 SCANSAR`  
  
