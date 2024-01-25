"""Code to preprocess a single FARSITE case"""

import numpy as np
import os
from helpers import Farsite2Google
import matplotlib.pyplot as pl
from shapely.geometry import shape
import fiona

rootPath = "D:/OneDrive - Imperial College London/Imperial/ME4/FYP/GeneratorFiles/bananaboat/"
moistureFiles = "fms"
burn_start = [2021,1,1,1500];       "Year, Month, Day, HHMM"
burn_duration = 3;                  "Hours"
steps_per_hour = 4;                 "15-min intervals"
cellSize = 20
xllcorner = 0
yllcorner = 0

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
                 yllcorner)
cover = Farsite2Google.get_asc_file(rootPath,'cover.asc')
slope_north, slope_east = FarsiteParams.get_slope_N_S_from_wxs()

#----------Landscape file, canopy-------------
height = Farsite2Google.get_asc_file(rootPath,'height.asc')
base = Farsite2Google.get_asc_file(rootPath,'base.asc')
density = Farsite2Google.get_asc_file(rootPath,'density.asc')

#---------Wind magnitude and direction----------

wind_north, wind_east = FarsiteParams.get_wind_N_S_from_wxs("bananaboat.wxs", burn_start, burn_duration, steps_per_hour)

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

pl.imshow(burnMap)

print("Loaded all matrices. Shape: ", np.shape(fuel))

#--------------Normalize Data and resize indices-----------------

print("Normalizing Data")

#norm_fuel does not need normalising, it does need recategorisation
norm_wind_east = FarsiteParams.expand_right_index(FarsiteParams.norm_data_by_mean_var(wind_east))
norm_wind_north = FarsiteParams.expand_right_index(FarsiteParams.norm_data_by_mean_var(wind_north))
norm_slope_east = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(slope_east))
norm_slope_north = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(slope_north))
norm_cover = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(cover))
norm_height = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(height))
norm_base = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(base))
norm_density = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(density))
norm_moisture_1 = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(moisture_1))
norm_moisture_10 = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(moisture_10))
norm_moisture_100 = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(moisture_100))
norm_moisture_her = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(moisture_her))
norm_moisture_woo = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.norm_data_by_mean_var(moisture_woo))

vegetation = FarsiteParams.expand_left_and_right_indeces(vegetation)
previous_front = FarsiteParams.expand_left_and_right_indeces(previous_front)
scar = FarsiteParams.expand_left_and_right_indeces(scar)

print("Rasters Normalized")
print("Reclassifying fuel map")

continuous_fuel_class = FarsiteParams.expand_left_and_right_indeces(FarsiteParams.reclassify_fuels_to_continuous(fuel))

print("Fuel map reclassification done")

#-----------------Prepare all channels----------------------

print("Preparing all channels")

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

channels_LSTM=np.ndarray((1,burn_duration*steps_per_hour,np.shape(fuel)[0],np.shape(fuel)[1],17));     "Prepare the 1 x T x H x W x 17 tensor"
channels_EPD=np.ndarray((1,np.shape(fuel)[0],np.shape(fuel)[1],17));                            "Prepare the 1 x H x W x 17 tensor"

front = FarsiteParams.expand_left_and_right_indeces(burnMap)
vegetation=vegetation - front
previous_front = front
scar = scar + front

channels_LSTM[0:0,0:0,:,:,0:3] = np.concatenate((vegetation,
                                    previous_front,
                                    scar),
                                        axis = -1)
channels_LSTM[0:0,0:0,:,:,3:3] = norm_wind_east[0,:,:,:]
channels_LSTM[0:0,0:0,:,:,4:4] = norm_wind_east[0,:,:,:]
channels_LSTM[0:0,0:0,:,:,5:17] = timestep_unchanging_channels

channels_EPD[0:0,:,:,0:3] = np.concatenate((vegetation,
                                    previous_front,
                                    scar),
                                        axis = -1)
channels_EPD[0:0,:,:,3:3] = norm_wind_east[0,:,:,:]
channels_EPD[0:0,:,:,4:4] = norm_wind_east[0,:,:,:]
channels_EPD[0:0,:,:,5:17] = timestep_unchanging_channels

print("Initial channels have been set up")

#--------------------------RUN THE MODEL ITSELF------------------------------

fire_evolution_EPD = np.ndarray(1,shape(fuel))

for i in range(1,8):
    """
    Run the EPD model 8 times to start the EPD model and 
    prepare inputs for conv_LSTM model
    """
    front = run_google_EPD_model(channels_EPD)
    
    channels_EPD[0:0,:,:,0:0] = channels_EPD[0:0,:,:,0:0] - front
    channels_EPD[0:0,:,:,1:1] = front
    channels_EPD[0:0,:,:,2:2] = channels_EPD[0:0,:,:,2:2] + front
    channels_EPD[0:0,:,:,3:3] = norm_wind_east[i,:,:,:]
    channels_EPD[0:0,:,:,4:4] = norm_wind_east[i,:,:,:]
    channels_EPD[0:0,:,:,5:17] = timestep_unchanging_channels
    
    channels_LSTM[0:0,i,:,:,0:5] = channels_EPD[0:0,:,:,0:5]
    
channels_LSTM_timestep=np.ndarray(1,1,shape(fuel),17)
    
for timestep in range(9,(burn_duration*steps_per_hour+1)):
    """
    Run both models concurrently
    
    EPD First
    """
    front_EPD = run_google_EPD_model(channels_EPD)
    
    channels_EPD[0:0,:,:,0:0] = channels_EPD[0:0,:,:,0:0] - front
    channels_EPD[0:0,:,:,1:1] = front
    channels_EPD[0:0,:,:,2:2] = channels_EPD[0:0,:,:,2:2] + front
    channels_EPD[0:0,:,:,3:3] = norm_wind_east[timestep,:,:,:]
    channels_EPD[0:0,:,:,4:4] = norm_wind_east[timestep,:,:,:]
    channels_EPD[0:0,:,:,5:17] = timestep_unchanging_channels
    
    """
    conv_LSTM
    """
    front_LSTM = run_google_LSTM_model(channels_LSTM)
    
    
    
    channels_LSTM_timestep[0:0,0:0,:,:,0:0] = channels_LSTM[0:0,8,:,:,0:0] - front
    channels_LSTM_timestep[0:0,0:0,:,:,1:1] = front
    channels_LSTM_timestep[0:0,0:0,:,:,2:2] = channels_LSTM[0:0,8,:,:,2:2] + front
    channels_LSTM_timestep[0:0,0:0,:,:,3:3] = norm_wind_east[timestep,:,:,:]
    channels_LSTM_timestep[0:0,0:0,:,:,4:4] = norm_wind_east[timestep,:,:,:]
    channels_LSTM_timestep[0:0,0:0,:,:,5:17] = timestep_unchanging_channels
    
    channels_LSTM
    


#tensor=tf.convert_to_tensor(np.array(channels_EPD),dtype=tf.float32)
#tf.saved_model.save(tensor, os.path.join(rootPath, "inputTensor.tsr"))



































    
    
