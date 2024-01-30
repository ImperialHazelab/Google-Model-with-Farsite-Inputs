"""Code to preprocess a single FARSITE case"""

import numpy as np
import os
from helpers import Farsite2Google
import matplotlib.pyplot as pl
from shapely.geometry import shape
import fiona
import runGoogleModels

rootPath = "D:\OneDrive - Imperial College London\Documents\Coding Projects\FireScenarioGenerator\FireScenarioGenerator\Output_windy/"
moistureFiles = "fms"
burn_start = [2024,1,1,1300];       "Year, Month, Day, HHMM"
burn_duration = 10;                  "Hours"
steps_per_hour = 4;                 "15-min intervals"
cellSize = 30
xllcorner = 0
yllcorner = 0

model = "multiFuel";  #Choose between singleFuel, multiFuel, california, california_wn

#----------Landscape file, surface-------------
#----------Create Object to save self variables-------------

fuel = Farsite2Google.get_asc_file(rootPath,'fuel.asc')
FarsiteParams = Farsite2Google(rootPath, 
                 moistureFiles, 
                 burn_start, 
                 burn_duration, 
                 steps_per_hour, 
                 np.shape(fuel),
                 cellSize,
                 xllcorner,
                 yllcorner,
                 model)
cover = Farsite2Google.get_asc_file(rootPath,'canopy.asc')
slope_north, slope_east = FarsiteParams.get_slope_N_S_from_wxs()

#----------Landscape file, canopy-------------
height = Farsite2Google.get_asc_file(rootPath,'height.asc')
base = Farsite2Google.get_asc_file(rootPath,'standheight.asc')
density = Farsite2Google.get_asc_file(rootPath,'crownbulkdensity.asc')

#---------Wind magnitude and direction----------

wind_north, wind_east = FarsiteParams.get_wind_N_S_from_wxs("weather.wxs", burn_start, burn_duration, steps_per_hour)

#---------------Moistures-----------------
if moistureFiles == "fms":

  moisture_1 = FarsiteParams.get_moisture_raster(fuel, "1hour")
  moisture_10 = FarsiteParams.get_moisture_raster(fuel, "10hour")
  moisture_100 = FarsiteParams.get_moisture_raster(fuel, "100hour")
  moisture_woo = FarsiteParams.get_moisture_raster(fuel, "woody")
  moisture_her = FarsiteParams.get_moisture_raster(fuel, "herbaceous")

if moistureFiles == "asc":
  moisture_1 = FarsiteParams.get_asc_file('moisture_1_hour.asc')
  moisture_10 = FarsiteParams.get_asc_file('moisture_10_hour.asc')
  moisture_100 = FarsiteParams.get_asc_file('moisture_100_hour.asc')
  moisture_her = FarsiteParams.get_asc_file('moisture_live_herbaceous.asc')
  moisture_woo = FarsiteParams.get_asc_file('moisture_live_woody.asc')
  
#----------Burn model outputs-------------

vegetation=np.ones(np.shape(fuel))
previous_front=np.zeros(np.shape(fuel))
scar=np.zeros(np.shape(fuel))

#----------------Ignition---------------


polygonIn=fiona.open(os.path.join(rootPath, "Ignition.shp"))
#first feature of the shapefile
first = polygonIn.next()
shp_geom = shape(first['geometry']); # or shp_geom = shape(first) with PyShp
burnMap = FarsiteParams.burnMap(shp_geom)

print("Loaded all matrices. Shape: ", np.shape(fuel))

#--------------Normalize Data and resize indices-----------------

print("=================Normalizing Data=================")

#norm_fuel does not need normalising, it does need recategorisation
norm_wind_east = FarsiteParams.expand_right_index(FarsiteParams.norm_data_by_norms(wind_east,4))
norm_wind_north = FarsiteParams.expand_right_index(FarsiteParams.norm_data_by_norms(wind_north,5))
norm_slope_east = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(slope_east,15))
norm_slope_north = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(slope_north,16))
norm_cover = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(cover,11))
norm_height = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(height,12))
norm_base = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(base,13))
norm_density = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(density,14))
norm_moisture_1 = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(moisture_1,6))
norm_moisture_10 = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(moisture_10,7))
norm_moisture_100 = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(moisture_100,8))
norm_moisture_her = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(moisture_her,9))
norm_moisture_woo = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_norms(moisture_woo,10))

vegetation = FarsiteParams.expand_left_and_right_indeces(vegetation)
previous_front = FarsiteParams.expand_left_and_right_indeces(previous_front)
scar = FarsiteParams.expand_left_and_right_indeces(scar)

print("====================Rasters Normalized======================")
print("======================Reclassifying fuel map======================")

continuous_fuel_class = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.reclassify_fuels_to_continuous(fuel))

print("======================Fuel map reclassification done======================")

#-----------------Prepare all channels----------------------

print("======================Preparing all channels======================")

timestep_unchanging_channels = np.concatenate((             
                                    norm_moisture_1,
                                    norm_moisture_10,
                                    norm_moisture_100,
                                    norm_moisture_her,
                                    norm_moisture_woo,
                                    norm_cover,
                                    norm_height,
                                    norm_base,
                                    norm_density,
                                    norm_slope_east,
                                    norm_slope_north,
                                    continuous_fuel_class),
                                    axis = -1);                     "All the channels that never change"

channels_LSTM=np.ndarray((1,8,np.shape(fuel)[0],np.shape(fuel)[1],17));     "Prepare the 1 x T x H x W x 17 tensor"
channels_EPD=np.ndarray((1,np.shape(fuel)[0],np.shape(fuel)[1],17));                            "Prepare the 1 x H x W x 17 tensor"

front = FarsiteParams.expand_left_and_right_indeces(burnMap)
vegetation=vegetation - front
previous_front = front
scar = scar + front

channels_LSTM[0,0,:,:,0] = vegetation[0,:,:,0]
channels_LSTM[0,0,:,:,1] = previous_front[0,:,:,0]
channels_LSTM[0,0,:,:,2] = scar[0,:,:,0]
channels_LSTM[0,0,:,:,3] = norm_wind_east[0,:,:,0]
channels_LSTM[0,0,:,:,4] = norm_wind_north[0,:,:,0]
channels_LSTM[0,0,:,:,5:17] = timestep_unchanging_channels

channels_EPD = channels_LSTM[0,0,:,:,:]
channels_EPD=FarsiteParams.expand_left_index(channels_EPD)

#FarsiteParams.channels2excel(channels_EPD[0,:,:,:],"channels_EPD.xlsx")

print("Initial channels have been set up")

#--------------------------RUN THE MODEL ITSELF------------------------------

fire_evolution_EPD = np.ndarray((1,np.shape(fuel)[0],np.shape(fuel)[1]))

pl.subplots(1,3)

for i in range(0,8):
    
    print("------ITERATION STEP ",i+1,"of ",burn_duration*steps_per_hour, " -----------")
    """
    Run the EPD model 8 times to start the EPD model and 
    prepare inputs for conv_LSTM model
    """
    front_EPD = runGoogleModels.run_google_EPD_model(channels_EPD,model)
    
    channels_EPD[0,:,:,0] = np.clip((channels_EPD[0,:,:,0] - front_EPD[0,:,:,0]),0,1)
    channels_EPD[0,:,:,1] = front_EPD[0,:,:,0]
    channels_EPD[0,:,:,2] = np.clip((channels_EPD[0,:,:,2] + front_EPD[0,:,:,0]),0,1)
    channels_EPD[0,:,:,3] = norm_wind_east[i,:,:,0]
    channels_EPD[0,:,:,4] = norm_wind_north[i,:,:,0]
        
    channels_LSTM[0,i,:,:,0:5] = channels_EPD[0,:,:,0:5]
    
    pl.subplot(1,3,1)
    pl.imshow(channels_EPD[0,:,:,0])
    pl.subplot(1,3,2)
    pl.imshow(channels_EPD[0,:,:,1])
    pl.subplot(1,3,3)
    pl.imshow(channels_EPD[0,:,:,2])
    
channels_LSTM_timestep=np.ndarray((1,1,np.shape(fuel)[0],np.shape(fuel)[1],17))

pl.subplots(2,3)
    
for timestep in range(8,(burn_duration*steps_per_hour)):
    
    print("------ITERATION STEP ",timestep+1,"of ",burn_duration*steps_per_hour, " -----------")
    """
    Run both models concurrently
    
    EPD First
    """
    front_EPD = runGoogleModels.run_google_EPD_model(channels_EPD,model)
    
    channels_EPD[0,:,:,0] = np.clip((channels_EPD[0,:,:,0] - front_EPD[0,:,:,0]),0,1)
    channels_EPD[0,:,:,1] = front_EPD[0,:,:,0]
    channels_EPD[0,:,:,2] = np.clip((channels_EPD[0,:,:,2] + front_EPD[0,:,:,0]),0,1)
    channels_EPD[0,:,:,3] = norm_wind_east[timestep,:,:,0]
    channels_EPD[0,:,:,4] = norm_wind_north[timestep,:,:,0]
    
    pl.subplot(2,3,1)
    pl.imshow(channels_EPD[0,:,:,0])
    pl.subplot(2,3,2)
    pl.imshow(channels_EPD[0,:,:,1])
    pl.subplot(2,3,3)
    pl.imshow(channels_EPD[0,:,:,2])
    
    """
    conv_LSTM
    """
    front_LSTM = runGoogleModels.run_google_LSTM_model(channels_LSTM,model)
    
    channels_LSTM_timestep[0,0,:,:,0] = np.clip((channels_LSTM[0,7,:,:,0] - front_LSTM[0,7,:,:,0]),0,1)
    channels_LSTM_timestep[0,0,:,:,1] = front_LSTM[0,7,:,:,0]
    channels_LSTM_timestep[0,0,:,:,2] = np.clip((channels_LSTM[0,7,:,:,2] + front_LSTM[0,7,:,:,0]),0,1)
    channels_LSTM_timestep[0,0,:,:,3] = norm_wind_east[timestep,:,:,0]
    channels_LSTM_timestep[0,0,:,:,4] = norm_wind_north[timestep,:,:,0]
    channels_LSTM_timestep[0,0,:,:,5:17] = timestep_unchanging_channels
    
    channels_LSTM = channels_LSTM[0,1:,:,:,:]
    channels_LSTM = np.concatenate((FarsiteParams.expand_left_index(channels_LSTM),channels_LSTM_timestep),axis=1)
    
    pl.subplot(2,3,4)
    pl.imshow(channels_LSTM[0,7,:,:,0])
    pl.subplot(2,3,5)
    pl.imshow(channels_LSTM[0,7,:,:,1])
    pl.subplot(2,3,6)
    pl.imshow(channels_LSTM[0,7,:,:,2])
    
    
    
print("=========Simulations Done===========")
    


#tensor=tf.convert_to_tensor(np.array(channels_EPD),dtype=tf.float32)
#tf.saved_model.save(tensor, os.path.join(rootPath, "inputTensor.tsr"))



































    
    
