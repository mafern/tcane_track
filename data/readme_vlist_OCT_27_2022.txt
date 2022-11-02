nnfit_vlist.dat file description. Last updated 10/26/2022

The nnfit_vlist_OCT_27_2022.dat file contains the input data used to train the NNIC and TCAN models. 
It is the output from the NNIC model training program for prediction of TC intensity from an ensemble of models and TC environmental input. 
The file is the input for the TCAN forecast uncertainty model training programs. Below is a description of the data in the nnfit_vlist.dat file. 

ATCF       4-letter storm identifier from the Automated Tropical Cyclone Forecast (ATCF) system. The first characters  are the basin ID and the 2nd two are the storm number for a given year. 
Name       Storm name
Date/time  Year mmddhh where mm=month,dd=day of the month, hh=hour (UTC) of the start of the forecast
ftime(hr)  Forecast time in hr
VMAX0      The max wind at the start of the forecast (kt)
NCI        The number of models included in the intensity consensus forecast
OFDV       The deviation of the NHC or CPHC official intensity forecast from the model consensus (kt)
OBDV       Same as OFDV but for the observed intensity from the best track 
DSDV       Same as OFDV but for the Decay-SHIPS intensity model forecast
LGDV       Same as OFDV but for the LGEM intensity model forecast
HWDV       Same as OFDV but for the early version of the HWRF intensity model forecast
DV12       The intensity change in the previous 12 hr from the start of the forecast (kt)
SLAT       The average latitude (deg N) from the DSHIPS forecast over the 24 h period up to the forecast time
SSTN       The average SST (deg C) at the TC center over the 24 h period up to the foreast time
SHDC       The average 850-200 hPa vertical shear (kt) in the TC environment averaged over the 24 h period up to the forecast time 
DTL        The distance of the TC center to the nearest major landmass (km) at the forecast time along the track used as input to the DSHIPS model
D200       The average 200 hPa divergence (1/sec x 10**5)  in the TC environment averaged over the 24 h period up to the forecast time 
T200       The average 200 hPa temperature (deg C)  in the TC environment averaged over the 24 h period up to the forecast time 
RHMD       The average 700-500 hPa layer humidity (percent) in the TC environment averaged over the 24 h period up to the forecast time 
SPDX       The average eastward-component of the TC motion vector (kt) from the DSHIPS track forecast averaged over tha 24 h period up to the forecast time 
SPDY       Same as SPDX for the northward component of the TC motion vector (kt)
NCT        The number of models included in the track consensus forecast
OFDX       The distance east (km) of the NHC or CPHC official track forecast from the consensus track forecast
OBDX       Same as OFDX but for the best track TC position
AVDX       Same as OFDX but for the early GFS global model track forecast
EMDX       Same as OFDX but for the early ECMWF global model track forecast
EGDX       Same as OFDX but for the early UKMet global model track forecast
HWDX       Same as OFDX but for the early HWRF global model track forecast
LONC       The longitude (0-360 deg convention) of the consensus track foreacst
OFDY,OBDY,AVDY,EMDY,EGDY,HWDY - Same as the DX variables above but for the northward displacement from the consensus track forecast
LATC       The latitude (deg N) of the consensus track forecast
