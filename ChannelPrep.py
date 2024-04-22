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
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from scipy.interpolate import griddata
from IPython import display 

class CallGoogleModel:
    
    def dist(A,B):
        return np.sum((np.sum((A-B)**2, axis=0))**0.5)
    
    def getRosFromArrivalTime (arrivalTime, cellSize) :
        gradients = np.gradient(arrivalTime[:,:,0], cellSize)
        #for i in range(0,np.shape(arrivalTime)[0]):
         #   for j in range(0,np.shape(arrivalTime)[1]):
          #      if gradients[0][i,j] != 0 :
           #         gradients[0][i,j] = 1 / gradients[0][i,j]
            #    if gradients[1][i,j] != 0 :
             #       gradients[1][i,j] = 1 / gradients[1][i,j]
        ROS = 1/np.sqrt(gradients[0][:,:]**2 + gradients[1][:,:]**2)
        
        return ROS
    

        
    
    def resolveArrivalTime(self, mode):
        if (mode=="EPD"):
            arrivalTime = np.copy(self.arrivalTime_EPD)
        elif (mode == "LSTM"):
            arrivalTime = np.copy(self.arrivalTime_LSTM)
            
        arrivalTime[self.channels_LSTM[0,0,:,:,2] != 0] = 1
        edgeNode = np.zeros(np.shape(arrivalTime))
        
        edgePointsList = []
        edgePointsArrivalTimes = []
        totalField = []
        for i in range(1,np.shape(arrivalTime)[0]-1):
            for j in range(1,np.shape(arrivalTime)[1]-1):
                edgeNode[i,j] = 0
                totalField.append([i,j])
                if arrivalTime[i-1,j] > arrivalTime[i,j] : edgeNode[i,j] = 1
                if arrivalTime[i,j-1] > arrivalTime[i,j] : edgeNode[i,j] = 1
                if arrivalTime[i+1,j] > arrivalTime[i,j] : edgeNode[i,j] = 1
                if arrivalTime[i,j+1] > arrivalTime[i,j] : edgeNode[i,j] = 1
                
                if arrivalTime[i-1,j] == 0 : edgeNode[i,j] = 1
                if arrivalTime[i,j-1] == 0 : edgeNode[i,j] = 1
                if arrivalTime[i+1,j] == 0 : edgeNode[i,j] = 1
                if arrivalTime[i,j+1] == 0 : edgeNode[i,j] = 1
                    
                if arrivalTime[i,j] == 0 : edgeNode[i,j] = 2
                
                if edgeNode[i,j] == 1 : 
                    edgePointsList.append([i,j])
                    edgePointsArrivalTimes.append(arrivalTime[i,j])
                        
        grid_z2 = griddata(np.array(edgePointsList), np.array(edgePointsArrivalTimes), np.asarray(totalField), method='cubic')
        
        gridOut = np.zeros([127,127])
        for i in range(np.shape(grid_z2)[0]):
            gridOut[totalField[i][0],totalField[i][1]]=grid_z2[i]
            
        gridOut[np.isnan(gridOut)] = 0
        
        return gridOut
    
    def resolveArrivalTime2(self, mode):
        
        if (mode=="EPD"):
            arrivalTime = np.copy(self.arrivalTime_EPD)
        elif (mode == "LSTM"):
            arrivalTime = np.copy(self.arrivalTime_LSTM)
            
        arrivalTime[self.channels_LSTM[0,0,:,:,2] != 0] = 1
        
        maxTime = np.max(arrivalTime)
        edgeNode = np.zeros(np.shape(arrivalTime))
        for i in range(1,np.shape(arrivalTime)[0]-1):
            for j in range(1,np.shape(arrivalTime)[1]-1):
                edgeNode[i,j] = 0
                if arrivalTime[i-1,j] > arrivalTime[i,j] : edgeNode[i,j] = 1
                if arrivalTime[i,j-1] > arrivalTime[i,j] : edgeNode[i,j] = 1
                if arrivalTime[i+1,j] > arrivalTime[i,j] : edgeNode[i,j] = 1
                if arrivalTime[i,j+1] > arrivalTime[i,j] : edgeNode[i,j] = 1
                
                if arrivalTime[i-1,j] == 0 : edgeNode[i,j] = 1
                if arrivalTime[i,j-1] == 0 : edgeNode[i,j] = 1
                if arrivalTime[i+1,j] == 0 : edgeNode[i,j] = 1
                if arrivalTime[i,j+1] == 0 : edgeNode[i,j] = 1
                    
                if arrivalTime[i,j] == 0 : edgeNode[i,j] = 2
        
        #return edgeNode        
        #arrivalTime[arrivalTime==0]=-9999
        
        for i in range(1,np.shape(arrivalTime)[0]-1):
            #print(100*i/(np.shape(arrivalTime)[0]-1))
            for j in range(1,np.shape(arrivalTime)[1]-1):
                point1=np.array([i,j])
                distToEachBoundary = {}
                if edgeNode[i,j] == 0 :
                    #Find the closest point of all the boundaries
                    for k in range(1,np.shape(arrivalTime)[0]-1):
                        for m in range(1,np.shape(arrivalTime)[1]-1):
                            point2 = np.array([k,m])
                            if edgeNode[k,m] == 1 :
                                pointDist = CallGoogleModel.dist(point1,point2)
                                if arrivalTime[k,m] not in distToEachBoundary:
                                    distToEachBoundary[arrivalTime[k,m]]=[pointDist,k,m]
                                else:
                                    if pointDist < distToEachBoundary[arrivalTime[k,m]][0]:
                                        distToEachBoundary[arrivalTime[k,m]] = [pointDist,k,m]
                    
                    sortedDistToEachNode = dict(sorted(distToEachBoundary.items(), key=lambda x:x[1]))
                    
                    #Choose the closest point, retain the boundaries before and after it
                    print(".,.,.,.,.,.,.,.,.,.,.,.,.,.,")
                    print(point1)
                    print(sortedDistToEachNode)
                    
                    if abs(list(sortedDistToEachNode.keys())[0] - list(sortedDistToEachNode.keys())[1]) != 15 :
                        edge1 = np.min([list(sortedDistToEachNode.keys())[0], list(sortedDistToEachNode.keys())[1]])
                    else:
                        edge1 = list(sortedDistToEachNode.keys())[0]
                    edge2 = edge1 + 15
                    edge3 = edge1 - 15
                    print("--------------------")
                    print(edge1)
                    print(edge2)
                    print(edge3)
                    print("--------------------")
                    node1 = sortedDistToEachNode[edge1]
                    if edge1 != maxTime : node2 = sortedDistToEachNode[edge2]
                    if edge1 != 15 : node3 = sortedDistToEachNode[edge3]
                    
                    #if the points are equidistant 
                    if edge1 == 15:
                        arrivalTime[i,j]=(edge1*node2[0]+edge2*node1[0])/(node1[0]+node2[0])
                    elif edge1 == maxTime:
                        arrivalTime[i,j]=(edge1*node3[0]+edge3*node1[0])/(node1[0]+node3[0])
                    elif node1[0] == node2[0]:
                        arrivalTime[i,j]=edge1+7.5
                    elif node1[0] == node3[0]:
                        arrivalTime[i,j]=edge1-7.5

                    else:
                        #Draw line between points 1-2 and 1-3, find if target point is within either of those lines
                        
                        # Point A: node1
                        # Point B: node2
                        # Point C: node3
                        # Point X: Target point
                        A = np.array([node1[1],node1[2]])
                        B = np.array([node2[1],node2[2]])
                        C = np.array([node3[1],node3[2]])
                        X = [i,j]
                        
                        AB = B - A
                        AC = C - A
                        AX = X - A
                        
                        D = A + AB * (np.dot(AB,AX) / np.dot(AB,AB))
                        E = A + AB * (np.dot(AC,AX) / np.dot(AC,AC))
                        
                        
                        
                        #Point is between edge1 and edge2
                        if CallGoogleModel.dist(A,D) + CallGoogleModel.dist(D,B) <= CallGoogleModel.dist(A,B) : 
                            if CallGoogleModel.dist(X,D) < CallGoogleModel.dist(X,E): 
                                arrivalTime[i,j]=(edge1*node2[0]+edge2*node1[0])/(node1[0]+node2[0])
                        #Point is between edge1 and edge3
                        #elif CallGoogleModel.dist(A,E) + CallGoogleModel.dist(E,C) <= CallGoogleModel.dist(A,C) : 
                        else:
                            arrivalTime[i,j]=(edge1*node3[0]+edge3*node1[0])/(node1[0]+node3[0])
                    print(arrivalTime[i,j])
                    print(".,.,.,.,.,.,.,.,.,.,.,.,.,.,")
                    #input()
        return arrivalTime
    
    def __init__(self, channels_EPD, channels_LSTM):
        
        self.channels_EPD = np.copy(channels_EPD)
        self.channels_LSTM = np.copy(channels_LSTM)
        self.arrivalTime_EPD = np.zeros(np.shape(channels_EPD[0,:,:,0]))
        self.arrivalTime_EPD[self.channels_EPD[0,:,:,1]>0]
        self.arrivalTime_LSTM = np.copy(self.arrivalTime_EPD)
        
    def iterate_EPD(self, dataset, timestep):
        front_EPD = runGoogleModels.run_google_EPD_model(Farsite2Google.expand_left_index(self.channels_EPD[timestep,:,:,:]), dataset)
        front_EPD = np.clip(front_EPD,0,1)
        self.channels_EPD[timestep+1,:,:,0] = np.clip((self.channels_EPD[timestep,:,:,0] - front_EPD[0,:,:,0]),0,1)
        self.channels_EPD[timestep+1,:,:,1] = front_EPD[0,:,:,0]
        self.channels_EPD[timestep+1,:,:,2] = np.clip((self.channels_EPD[timestep,:,:,2] + front_EPD[0,:,:,0]),0,1)
        
        self.arrivalTime_EPD[np.logical_and(front_EPD[0,:,:,0] > 0.1, self.arrivalTime_EPD == 0)] = 15 * timestep
        
    def iterate_LSTM(self, dataset, timestep):
        
        front_LSTM = runGoogleModels.run_google_LSTM_model(Farsite2Google.expand_left_index(self.channels_LSTM[0,timestep:timestep+8,:,:,:]),dataset)
        front_LSTM = np.clip(front_LSTM,0,1)
        self.channels_LSTM[0,timestep+8,:,:,0] = np.clip((self.channels_LSTM[0,timestep+7,:,:,0] - front_LSTM[0,7,:,:,0]),0,1)
        self.channels_LSTM[0,timestep+8,:,:,1] = front_LSTM[0,7,:,:,0]
        self.channels_LSTM[0,timestep+8,:,:,2] = np.clip((self.channels_LSTM[0,timestep+7,:,:,2] + front_LSTM[0,7,:,:,0]),0,1)
        
        self.arrivalTime_LSTM[np.logical_and(front_LSTM[0,7,:,:,0] > 0.1, self.arrivalTime_LSTM == 0)] = 15 * (timestep+1)
             
    def plotResults(self, timestep, label, name, path):
        
        fig, axs = pl.subplots(nrows=3, ncols=2, figsize=(8, 8), tight_layout=True)
        
        plot = axs[0, 0].imshow(self.channels_EPD[timestep,:,:,0],cmap="plasma")
        axs[0, 0].set_title(("EPD model"))
        pl.colorbar(plot)
        axs[0, 0].axis("on")
        
        plot = axs[0, 1].imshow(self.channels_LSTM[0,timestep+8,:,:,0],cmap="plasma")
        axs[0, 1].set_title("LSTM model")
        pl.colorbar(plot)
        axs[0, 1].axis("on")
        
        plot = axs[1, 0].imshow(self.channels_EPD[timestep,:,:,0]-label,cmap="coolwarm")
        axs[1, 0].set_title("EPD error")
        pl.colorbar(plot)
        axs[1, 0].axis("off")
        
        plot = axs[1, 1].imshow(self.channels_LSTM[0,timestep+8,:,:,0]-label,cmap="coolwarm")
        axs[1, 1].set_title("LSTM error")
        pl.colorbar(plot)
        axs[1, 1].axis("off")
        
        plot = axs[2, 0].imshow(self.arrivalTime_EPD, cmap = "Greens")
        axs[2, 0].set_title("EPD Arrival Time")
        pl.colorbar(plot)
        axs[2, 0].axis("off")
        
        plot = axs[2, 1].imshow(self.arrivalTime_LSTM, cmap = "Greens")
        axs[2, 1].set_title("LSTM Arrival Time")
        pl.colorbar(plot)
        axs[2, 1].axis("off")
        
        pl.suptitle(name)
        pl.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        
        pl.savefig(path + name + ".png")
        
        pl.show()
    
    def examineDifference(timestep,label,LSTM):
        fig, axs = pl.subplots(nrows=3, ncols=1, tight_layout=True)
        
        plot = axs[0].imshow(LSTM[timestep,:,:],cmap="plasma")
        axs[0].set_title(("LSTM model"))
        pl.colorbar(plot)
        axs[0].axis("on")
        
        plot = axs[1].imshow(label[:,:,timestep],cmap="coolwarm")
        axs[1].set_title("Label")
        pl.colorbar(plot)
        axs[1].axis("off")
        
        plot = axs[2].imshow(label[:,:,timestep]-LSTM[timestep,:,:], cmap = "Greens")
        axs[2].set_title("Difference")
        pl.colorbar(plot)
        axs[2].axis("off")
        
        pl.suptitle("timestep " + str(timestep) + " difference")
        
        pl.show()
        
    def animateResults(self, channel, quantity, cmap='viridis', interval=200):
        fig, ax = pl.subplots()
        
        if (channel == "EPD"):
            matrix_data = self.channels_EPD
        elif (channel == "LSTM"):
            matrix_data = self.channels_LSTM[0,7:,:,:,:]
        else:
            print("Model specified not in the list")
            return 0;
        
        if (quantity == "vegetation"):
            quantity = 0
        elif (quantity == "front"):
            quantity = 1
        elif (quantity == "scar"):
            quantity = 2
        
        img = ax.imshow(matrix_data[0, :, : , quantity], cmap=cmap)
        
        def update(i):
            img.set_array(matrix_data[i, :, : , quantity])
            ax.set_title("Model: " + channel + ', Time Step: ' + str(i))
            return img,
        
        num_frames = matrix_data.shape[1]
        animation = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
        
        pl.show()
        
        return animation

    def animateDifference(difference, cmap='viridis', interval=100):
        
        fig = pl.figure()
        im = pl.imshow(difference[0,:,:], vmin=0, vmax=2, interpolation='none', aspect='auto')
        pl.colorbar(im)
        
        def animate_func(i):        
            im.set_array(difference[i,:,:])
            return [im]
        
        anim = FuncAnimation(
                            fig, 
                            animate_func, 
                            frames = np.shape(difference)[0],
                            interval = interval, # in ms
                            )
        
        anim.save('test_anim.mp4', fps=3, extra_args=['-vcodec', 'libx264'])
        
        return anim    
                
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
        self.simSteps = self.burn_duration * self.steps_per_hour ;
        
    def importFiles(self):

        self.fuel = Farsite2Google.get_asc_file(self.rootPath,'fuel.asc')
        self.FarsiteParams = Farsite2Google(self.rootPath, 
                         self.moistureFiles,
                         self.burn_duration, 
                         self.steps_per_hour, 
                         np.shape(self.fuel),
                         self.cellSize,
                         self.xllcorner,
                         self.yllcorner,
                         self.simSteps)
        self.cover = Farsite2Google.get_asc_file(self.rootPath,'canopy.asc')
        self.slope_north, self.slope_east = self.FarsiteParams.get_slope_N_S_from_wxs()
        
        #----------Landscape file, canopy-------------
        self.height = Farsite2Google.get_asc_file(self.rootPath,'standheight.asc')*100
        self.base = Farsite2Google.get_asc_file(self.rootPath,'height.asc')*100
        self.density = Farsite2Google.get_asc_file(self.rootPath,'crownbulkdensity.asc')
        
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
        
    def exportRawValues(self):
        outputRaw = np.stack((self.vegetation,
                                self.previous_front,
                                self.scar,
                                self.wind_east[0,:,:],
                                self.wind_north[0,:,:],
                                self.moisture_1,
                                self.moisture_10,
                                self.moisture_100,
                                self.moisture_her,
                                self.moisture_woo,
                                self.cover,
                                self.height,
                                self.base,
                                self.density,
                                self.slope_east,
                                self.slope_north,
                                self.FarsiteParams.reclassify_fuels_to_continuous(self.fuel)),
                                axis = -1);                     "All the channels that never change"
        
        
        self.FarsiteParams.channels2excel(outputRaw[:,:,:],"channels_raw.xlsx")
          
    def normaliseAndStitchChannels(self, model, exportToExcel):
  
        #--------------Normalize Data and resize indices-----------------

        print("====================== Normalizing Data ========================")
        
        if (model == "california" or model == "california_wn"):
            self.height = self.height / 10
            self.base = self.base / 10
            self.density = self.density * 100
        
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
        
        channels_LSTM = np.ndarray((1,self.simSteps+7,np.shape(self.fuel)[0],np.shape(self.fuel)[1],17));     "Prepare the 1 x T x H x W x 17 tensor"
        channels_EPD = np.ndarray((self.simSteps,np.shape(self.fuel)[0],np.shape(self.fuel)[1],17));                            "Prepare the 1 x H x W x 17 tensor"
        
        
        for i in range(0, 8):
            channels_LSTM[0,i,:,:,0] = vegetation[0,:,:,0]
            channels_LSTM[0,i,:,:,1] = previous_front[0,:,:,0]
            channels_LSTM[0,i,:,:,2] = scar[0,:,:,0]
            channels_LSTM[0,i,:,:,3] = norm_wind_east[0,:,:,0]
            channels_LSTM[0,i,:,:,4] = norm_wind_north[0,:,:,0]
            channels_LSTM[0,i,:,:,5:17] = timestep_unchanging_channels
            
        for i in range(8, np.shape(channels_LSTM[0,:,0,0,0])[0]):
            channels_LSTM[0,i,:,:,0] = vegetation[0,:,:,0]
            channels_LSTM[0,i,:,:,1] = previous_front[0,:,:,0]
            channels_LSTM[0,i,:,:,2] = scar[0,:,:,0]
            channels_LSTM[0,i,:,:,3] = norm_wind_east[i-8,:,:,0]
            channels_LSTM[0,i,:,:,4] = norm_wind_north[i-8,:,:,0]
            channels_LSTM[0,i,:,:,5:17] = timestep_unchanging_channels
        
        front = Farsite2Google.expand_left_and_right_indeces(self.burnMap)
        vegetation = vegetation - front
        previous_front = front
        scar = scar + front
        
        channels_LSTM[0,7,:,:,0] = vegetation[0,:,:,0]
        channels_LSTM[0,7,:,:,1] = previous_front[0,:,:,0]
        channels_LSTM[0,7,:,:,2] = scar[0,:,:,0]
        
        channels_EPD = np.copy(channels_LSTM[0,:,:,:,:])
        
        channels_EPD_resized = np.ndarray((self.simSteps,126,126,17));
        channels_LSTM_resized = np.ndarray((1,self.simSteps+7,126,126,17));
        
        for i in range(0,self.simSteps):
            channels_EPD_resized[i,:,:,:] = tf.image.resize(channels_EPD[i,:,:,:], (126, 126),"nearest")
            channels_LSTM_resized[0,i,:,:,:] = tf.image.resize(channels_LSTM[0,i,:,:,:], (126, 126),"nearest")
        for i in range(0,7):
            channels_LSTM_resized[0,self.simSteps+i,:,:,:] = tf.image.resize(channels_LSTM[0,self.simSteps+i,:,:,:], (126, 126),"nearest")
            
        if exportToExcel:
            self.FarsiteParams.channels2excel(channels_EPD_resized[0,:,:,:],"channels_EPD.xlsx")

        print("====================== Initial Channel Setup Complete ==========")
        
        return channels_EPD_resized, channels_LSTM_resized


        
        