# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:24:31 2024

@author: nikos
"""

import os
from helpers import Farsite2Google
from shapely.geometry import shape
import fiona
import runGoogleModels
import numpy as np
import matplotlib.pyplot as pl

class CallGoogleModel:
    
    def __init__(self, channels_EPD, channels_LSTM):
        
        self.channels_EPD = channels_EPD
        self.channels_LSTM = channels_LSTM
        self.arrivalTime_EPD = np.zeros(np.shape(channels_EPD[0,:,:,0]))
        self.arrivalTime_LSTM = self.arrivalTime_EPD
        
    def iterate_EPD(self, dataset, timestep):
        front_EPD = runGoogleModels.run_google_EPD_model(Farsite2Google.expand_left_index(self.channels_EPD[timestep,:,:,:]), dataset)
        
        self.channels_EPD[timestep+1,:,:,0] = np.clip((self.channels_EPD[timestep,:,:,0] - front_EPD[0,:,:,0]),0,1)
        self.channels_EPD[timestep+1,:,:,1] = front_EPD[0,:,:,0]
        self.channels_EPD[timestep+1,:,:,2] = np.clip((self.channels_EPD[timestep,:,:,2] + front_EPD[0,:,:,0]),0,1)
        
        self.arrivalTime_EPD[np.logical_and(front_EPD[0,:,:,0] > 0.2, self.arrivalTime_EPD == 0)] = 15 * timestep
    def iterate_LSTM(self,dataset, timestep):
        
        front_LSTM = runGoogleModels.run_google_LSTM_model(Farsite2Google.expand_left_index(self.channels_LSTM[0,timestep-8:timestep,:,:,:]),dataset)
        
        self.channels_LSTM[0,timestep+1,:,:,0] = np.clip((self.channels_LSTM[0,timestep,:,:,0] - front_LSTM[0,7,:,:,0]),0,1)
        self.channels_LSTM[0,timestep+1,:,:,1] = front_LSTM[0,7,:,:,0]
        self.channels_LSTM[0,timestep+1,:,:,2] = np.clip((self.channels_LSTM[0,timestep,:,:,2] + front_LSTM[0,7,:,:,0]),0,1)
        
        self.arrivalTime_LSTM[np.logical_and(front_LSTM[0,7,:,:,0] > 0.2, self.arrivalTime_LSTM == 0)] = 15 * timestep
             
    def plotResults(self, timestep, label, name, path):
        
        fig, axs = pl.subplots(nrows=3, ncols=2, figsize=(8, 8), tight_layout=True)
        
        plot = axs[0, 0].imshow(self.channels_EPD[timestep,:,:,0],cmap="plasma")
        axs[0, 0].set_title(("EPD model"))
        pl.colorbar(plot)
        axs[0, 0].axis("off")
        
        plot = axs[0, 1].imshow(self.channels_LSTM[0,timestep,:,:,0],cmap="plasma")
        axs[0, 1].set_title("LSTM model")
        pl.colorbar(plot)
        axs[0, 1].axis("off")
        
        plot = axs[1, 0].imshow(self.channels_EPD[timestep,:,:,0]-label,cmap="coolwarm")
        axs[1, 0].set_title("EPD error")
        pl.colorbar(plot)
        axs[1, 0].axis("off")
        
        plot = axs[1, 1].imshow(self.channels_LSTM[0,timestep,:,:,0]-label,cmap="coolwarm")
        axs[1, 1].set_title("LSTM error")
        pl.colorbar(plot)
        axs[1, 1].axis("off")
        
        plot = axs[2, 0].imshow(self.arrivalTime_EPD, cmap = "Greens")
        axs[2, 0].set_title("EPD Arrival Time")
        pl.colorbar(plot)
        axs[2, 0].axis("off")
        
        plot = axs[2, 1].imshow(self.arrivalTime_EPD, cmap = "Greens")
        axs[2, 1].set_title("LSTM Arrival Time")
        pl.colorbar(plot)
        axs[2, 1].axis("off")
        
        pl.suptitle(name)
        pl.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        
        pl.savefig(path + name + ".png")
        
        pl.show()
                
class ChannelPrep:
    def __init__(self, config):
        
        self.rootPath = config[0][1]
        self.burn_start = config[1][1]
        self.burn_duration = config[2][1]
        self.steps_per_hour = config[3][1]
        self.cellSize = config[4][1]
        self.moistureFiles = config[5][1]
        self.xllcorner = config[6][1]
        self.yllcorner = config[7][1]
        
    def importFiles(self):

        self.fuel = Farsite2Google.get_asc_file(self.rootPath,'fuel.asc')
        self.FarsiteParams = Farsite2Google(self.rootPath, 
                         self.moistureFiles,
                         self.burn_duration, 
                         self.steps_per_hour, 
                         np.shape(self.fuel),
                         self.cellSize,
                         self.xllcorner,
                         self.yllcorner)
        self.cover = Farsite2Google.get_asc_file(self.rootPath,'canopy.asc')
        self.slope_north, self.slope_east = self.FarsiteParams.get_slope_N_S_from_wxs()
        
        #----------Landscape file, canopy-------------
        self.height = Farsite2Google.get_asc_file(self.rootPath,'height.asc')*10
        self.base = Farsite2Google.get_asc_file(self.rootPath,'standheight.asc')*100
        self.density = Farsite2Google.get_asc_file(self.rootPath,'crownbulkdensity.asc')*100
        
        #---------Wind magnitude and direction----------
        
        self.wind_north, self.wind_east = self.FarsiteParams.get_wind_N_S_from_wxs(self.burn_start, "weather.wxs")
        
        #---------------Moistures-----------------
        if self.moistureFiles == "fms":
        
          self.moisture_1 = self.FarsiteParams.get_moisture_raster(self.fuel, "1hour")
          self.moisture_10 = self.FarsiteParams.get_moisture_raster(self.fuel, "10hour")
          self.moisture_100 = self.FarsiteParams.get_moisture_raster(self.fuel, "100hour")
          self.moisture_woo = self.FarsiteParams.get_moisture_raster(self.fuel, "woody")
          self.moisture_her = self.FarsiteParams.get_moisture_raster(self.fuel, "herbaceous")
        
        if self.moistureFiles == "asc":
          self.moisture_1 = self.FarsiteParams.get_asc_file('moisture_1_hour.asc')
          self.moisture_10 = self.FarsiteParams.get_asc_file('moisture_10_hour.asc')
          self.moisture_100 = self.FarsiteParams.get_asc_file('moisture_100_hour.asc')
          self.moisture_her = self.FarsiteParams.get_asc_file('moisture_live_herbaceous.asc')
          self.moisture_woo = self.FarsiteParams.get_asc_file('moisture_live_woody.asc')
          
          
        self.vegetation=np.ones(np.shape(self.fuel))
        self.previous_front=np.zeros(np.shape(self.fuel))
        self.scar=np.zeros(np.shape(self.fuel))
  
        #----------------Ignition---------------
  
  
        polygonIn=fiona.open(os.path.join(self.rootPath, "Ignition.shp"))
        #first feature of the shapefile
        first = polygonIn.next()
        shp_geom = shape(first['geometry']); # or shp_geom = shape(first) with PyShp
        self.burnMap = self.FarsiteParams.burnMap(shp_geom)
  
        print("Loaded all matrices. Shape: ", np.shape(self.fuel))
          
    def normaliseAndStitchChannels(self, model, exportToExcel):
  
        #--------------Normalize Data and resize indices-----------------

        print("====================== Normalizing Data ========================")
        
        #norm_fuel does not need normalising, it does need recategorisation
        norm_wind_east = Farsite2Google.expand_right_index(self.FarsiteParams.norm_data_by_norms(self.wind_east, model, 4))
        norm_wind_north = Farsite2Google.expand_right_index(self.FarsiteParams.norm_data_by_norms(self.wind_north, model, 5))
        norm_slope_east = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.slope_east, model, 15))
        norm_slope_north = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.slope_north, model, 16))
        norm_cover = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.cover, model, 11))
        norm_height = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.height, model, 12))
        norm_base = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.base, model, 13))
        norm_density = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.density, model, 14))
        norm_moisture_1 = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.moisture_1, model, 6))
        norm_moisture_10 = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.moisture_10, model, 7))
        norm_moisture_100 = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.moisture_100, model, 8))
        norm_moisture_her = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.moisture_her, model, 9))
        norm_moisture_woo = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.norm_data_by_norms(self.moisture_woo, model, 10))
        
        vegetation = Farsite2Google.expand_left_and_right_indeces(self.vegetation)
        previous_front = Farsite2Google.expand_left_and_right_indeces(self.previous_front)
        scar = Farsite2Google.expand_left_and_right_indeces(self.scar)
        
        print("====================== Rasters Normalized ======================")
        print("====================== Reclassifying fuel map ==================")
        
        continuous_fuel_class = Farsite2Google.expand_left_and_right_indeces(self.FarsiteParams.reclassify_fuels_to_continuous(self.fuel))
        
        print("====================== Fuel map reclassification done ==========")

#-----------------Prepare all channels----------------------
        
        print("====================== Preparing all channels ==================")
        
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
        
        channels_LSTM=np.ndarray((1,self.steps_per_hour*self.burn_duration,np.shape(self.fuel)[0],np.shape(self.fuel)[1],17));     "Prepare the 1 x T x H x W x 17 tensor"
        channels_EPD=np.ndarray((self.steps_per_hour*self.burn_duration,np.shape(self.fuel)[0],np.shape(self.fuel)[1],17));                            "Prepare the 1 x H x W x 17 tensor"
        
        front = Farsite2Google.expand_left_and_right_indeces(self.burnMap)
        vegetation=vegetation - front
        previous_front = front
        scar = scar + front
        
        channels_LSTM[0,:,:,:,0] = vegetation[0,:,:,0]
        channels_LSTM[0,:,:,:,1] = previous_front[0,:,:,0]
        channels_LSTM[0,:,:,:,2] = scar[0,:,:,0]
        channels_LSTM[0,:,:,:,3] = norm_wind_east[0,:,:,0]
        channels_LSTM[0,:,:,:,4] = norm_wind_north[0,:,:,0]
        channels_LSTM[0,:,:,:,5:17] = timestep_unchanging_channels
        
        channels_EPD = channels_LSTM[0,:,:,:,:]
        
        if exportToExcel:
            self.FarsiteParams.channels2excel(channels_EPD[0,:,:,:],"channels_EPD.xlsx")

        print("====================== Initial Channel Setup Complete ==========")
        
        return channels_EPD, channels_LSTM


        
        