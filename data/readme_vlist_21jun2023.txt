nnfit_vlist.dat file description
Last updated 06/21/2023
Generated from v4.0.1 of nnfit.f90
This version used the final best tracks from the 2022 hurricane season. 
Header line changed and variable spacing modified 

The nnfit_vlist.dat file contains the input data used to train the NNIC and TCANE models. 
It is the output from the NNIC model traing program for prediction of TC intensity from an ensemble of models and TC environmental input. 
The fle is the input for the TCANE forecast uncertainty model training programs. Below is a description of the data in the nnfit_vlist.dat file. 

ATCFID     8-character storm identifier from the Automated Tropical Cyclone Forecast (ATCF) system.  
              The first two characters are the basin ID and the 2nd two are the storm number for a given year, followed by the year. 
NAME       Storm name
YEAR       Year of the forecast initialization time (can be different than the year in the ATCFID if a TC starts near the end of the year)
MMDDHH     Date/time of the forecast where MM=month,DD=day of the month, HH=hour (UTC) of the start of the forecast
FHOUR      Forecast time in hr
VMAX0      The max wind at the start of the forecast (kt)
NCI        The number of models included in the intensity consensus forecast
VMAXN      The max wind (kt) of the NHC or CPHC official forecast
OFDV       The deviation of the best track intensity from the NHC or CPHC official intensity foreacst (kt) (best track - official)
OBDV       Deviation of the best track intensity from the consensus                                        (best track - consensus)
DSDV       Deviation but for the Decay-SHIPS intensity model forecast from the consensus                   (DSHIPS - consensus)
LGDV       Same as DSDV but for the LGEM intensity model forecast                                          (LGEM - consensus)
HWDV       Same as DSDV but for the early version of the HWRF intensity model forecast                     (HWFI - consensus)
DV12       The intensity change in the previous 12 hr from the start of the forecast (kt)                  (Vmax(t=0) - Vmax(t=-12h)
SLAT       The average latitude (deg N) from the DSHIPS forecast over the 24 h period up to the forecast time
SSTN       The average SST (deg C) at the TC center over the 24 h period up to the foreast time
SHDC       The average 850-200 hPa vertical shear (kt) in the TC environment averaged over the 24 h period up to the forecast time 
DTL        The distance of the TC center to the nearest major landmass (km) at the forecast time along the track used as input to the DSHIPS model
D200       The average 200 hPa divergence (1/sec x 10**5)  in the TC environment averaged over the 24 h period up to the forecast time 
T200       The average 200 hPa temperature (deg C)  in the TC environment averaged over the 24 h period up to the forecast time 
RHMD       The average 700-500 hPa layer humidity (percent) in the TC environment averaged over the 24 h period up to the forecast time 
SPDX       The average eastward-component of the TC motion vector (kt) from the DSHIPS track forecast averaged over tha 24 h period up to the forecast time 
SPDY       Same as SPDX for the northward component of the TC motion vector (kt)
LON0       The longitude at the start of the forecast (0-360 deg)
LAT0       The latitude at the start of the forecast (deg N)
NCT        The number of models included in the track consensus forecast
LONN       The longitude of the NHC or CPHC official forecast (0-360 deg)
LATN       The latitude of the NHC or CPHC official forecast (deg N)
OFDX       The distance east (km) of the best track position from the NHC or CPHC official track forecast        (best track - official)
OBDX       The distance east (km) of the best track position from the consensus forecast                         (best track - consensus) 
AVDX       The distance east (km) of the early GFS global model track forecast from the consensus track forecast (GFSI - consensus)
EMDX       Same as AVDX but for the early ECMWF global model track forecast                                      (EMXI - consensus)
EGDX       Same as AVDX but for the early UKMet global model track forecast                                      (EGRI - consensus)
HWDX       Same as AVDX but for the early HWRF global model track forecast                                       (HWFI - consensus)
LONC       The longitude (0-360 deg convention) of the consensus track foreacst
OFDY       The distance north (km) of the best track position from the NHC or CPHC official track forecast        (best track - official)
OBDY       The distance north (km) of the best track position from the consensus forecast                         (best track - consensus) 
AVDY       The distance north (km) of the early GFS global model track forecast from the consensus track forecast (GFSI - consensus)
EMDY       Same as AVDY but for the early ECMWF global model track forecast                                       (EMXI - consensus)
EGDY       Same as AVDY but for the early UKMet global model track forecast                                       (EGRI - consensus)
HWDY       Same as AVDY but for the early HWRF global model track forecast                                        (HWFI - consensus)
LATC       The latitude (deg N) of the consensus track forecast
