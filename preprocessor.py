"""Code to preprocess a single FARSITE case"""
import numpy as np
from ChannelPrep import ChannelPrep
from ChannelPrep import CallGoogleModel
from helpers import Farsite2Google

rootPath = "D:\OneDrive - Imperial College London\Documents\Coding Projects\FireScenarioGenerator\FireScenarioGenerator\Output_complex2/"
moistureFiles = "fms"
burn_start = [2024,1,1,1300];       "Year, Month, Day, HHMM"
burn_duration = 24;                  "Hours"
steps_per_hour = 4;                 "15-min intervals"
cellSize = 30
xllcorner = 0
yllcorner = 0

#----------Create Object to save self variables-------------

config=[["rootpath",rootPath],
        ["burn start", burn_start],
        ["burn duration", burn_duration],
        ["steps per hour", steps_per_hour],
        ["cell size", cellSize],
        ["Moisture Channel Origin", "fms"],
        ["X Lower Left Corner Coords", xllcorner],
        ["Y Lower Left Corner Coords", yllcorner]]

prep = ChannelPrep(config)

label = Farsite2Google.get_asc_file(rootPath, "arrivaltime.asc")
label[label > 0] = 0
label[label < 0] = 1

prep.importFiles()

rasterSize=np.shape(prep.fuel)

EPD, LSTM = prep.normaliseAndStitchChannels("singleFuel", True)
simpleFuel = CallGoogleModel(EPD, LSTM)

EPD, LSTM = prep.normaliseAndStitchChannels("multiFuel", False)
multiFuel = CallGoogleModel(EPD, LSTM)

EPD, LSTM = prep.normaliseAndStitchChannels("california", False)
california = CallGoogleModel(EPD, LSTM)

EPD, LSTM = prep.normaliseAndStitchChannels("california_wn", False)
california_wn = CallGoogleModel(EPD, LSTM)

#--------------------------RUN THE MODEL ITSELF------------------------------

fire_evolution_EPD = np.ndarray((1,rasterSize[0],rasterSize[1]))

for i in range(0,8):
    
    print("------ITERATION STEP ",i+1,"of ",burn_duration*steps_per_hour, " -----------")
    """
    Run the EPD model 8 times to start the EPD model and 
    prepare inputs for conv_LSTM model
    """
    simpleFuel.iterate_EPD("singleFuel",i)
    simpleFuel.channels_LSTM[0,i,:,:,0:5] = simpleFuel.channels_EPD[0,:,:,0:5]
    
    multiFuel.iterate_EPD("multiFuel",i)
    multiFuel.channels_LSTM[0,i,:,:,0:5] = multiFuel.channels_EPD[0,:,:,0:5]
    
    california.iterate_EPD("california",i)
    california.channels_LSTM[0,i,:,:,0:5] = california.channels_EPD[0,:,:,0:5]
    
    california_wn.iterate_EPD("california_wn",i)
    california_wn.channels_LSTM[0,i,:,:,0:5] = california_wn.channels_EPD[0,:,:,0:5]

    
for timestep in range(8,(burn_duration*steps_per_hour)-1):
    
    print("------ITERATION STEP ",timestep+1,"of ",burn_duration*steps_per_hour, " -----------")
    """
    Run all models concurrently
    
    """
    
    simpleFuel.iterate_EPD("singleFuel",timestep)
    multiFuel.iterate_EPD("multifuel",timestep)
    california.iterate_EPD("california",timestep)
    california_wn.iterate_EPD("california_wn",timestep)
    simpleFuel.iterate_LSTM("singleFuel",timestep)
    multiFuel.iterate_LSTM("multiFuel",timestep)
    california.iterate_LSTM("california",timestep)
    california_wn.iterate_LSTM("california_wn",timestep)
  

simpleFuel.plotResults(timestep, label, "Single Fuel", rootPath)
multiFuel.plotResults(timestep, label, "Multiple Fuel", rootPath)
california.plotResults(timestep, label, "California", rootPath)
california_wn.plotResults(timestep, label, "California WN", rootPath)

    
print("=========Simulations Done===========")
