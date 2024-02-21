"""Code to preprocess a single FARSITE case"""
import numpy as np
from ChannelPrep import ChannelPrep
from ChannelPrep import CallGoogleModel
from helpers import Farsite2Google
import tensorflow as tf

rootPath = "D:\GoogleModel\wildfire_conv_ltsm\InputFiles"
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
label[label>0]=0;
label[label<0]=1

rasterSize=np.shape(label)
resize_ratio = 126/rasterSize[0]
prep.simSteps = np.round(burn_duration * steps_per_hour * resize_ratio).astype(int)

prep.importFiles()

rasterSize=np.shape(prep.fuel)
resize_ratio = 126/rasterSize[0]
prep.simSteps = np.round(burn_duration * steps_per_hour * resize_ratio).astype(int)
 
# prep.exportRawValues()

EPD, LSTM = prep.normaliseAndStitchChannels("singleFuel", False)
simpleFuel = CallGoogleModel(EPD, LSTM)

EPD, LSTM = prep.normaliseAndStitchChannels("multiFuel", False)
multiFuel = CallGoogleModel(EPD, LSTM)

EPD, LSTM = prep.normaliseAndStitchChannels("california", False)
california = CallGoogleModel(EPD, LSTM)

EPD, LSTM = prep.normaliseAndStitchChannels("california_wn", False)
california_wn = CallGoogleModel(EPD, LSTM)

#--------------------------ADJUST TIME SCALE--------------------------------



#--------------------------RUN THE MODEL ITSELF------------------------------

fire_evolution_EPD = np.ndarray((1,rasterSize[0],rasterSize[1]))

for i in range(0,8):
    
    print("------ITERATION STEP ",i+1,"of ",prep.simSteps, " -----------")
    """
    Run the EPD model 8 times to start the EPD model and 
    prepare inputs for conv_LSTM model
    """
    # simpleFuel.iterate_EPD("singleFuel",i)
    # multiFuel.iterate_EPD("multiFuel",i)
    california.iterate_EPD("california",i)
    # california_wn.iterate_EPD("california_wn",i)
    
    
# simpleFuel.channels_LSTM[0,0:9,:,:,0:3] = np.copy(simpleFuel.channels_EPD[0:9,:,:,0:3])
# multiFuel.channels_LSTM[0,0:9,:,:,0:3] = np.copy(multiFuel.channels_EPD[0:9,:,:,0:3])
california.channels_LSTM[0,0:9,:,:,0:3] = np.copy(california.channels_EPD[0:9,:,:,0:3])
# california_wn.channels_LSTM[0,0:9,:,:,0:3] = np.copy(california_wn.channels_EPD[0:9,:,:,0:3])

# simpleFuel.arrivalTime_LSTM = np.copy(simpleFuel.arrivalTime_EPD)
# multiFuel.arrivalTime_LSTM = np.copy(multiFuel.arrivalTime_EPD)
california.arrivalTime_LSTM = np.copy(california.arrivalTime_EPD)
# california_wn.arrivalTime_LSTM = np.copy(california_wn.arrivalTime_EPD)
    
for timestep in range(8,prep.simSteps-1):
    
    print("------ITERATION STEP ",timestep+1,"of ",prep.simSteps, " -----------")
    """
    Run all models concurrently
    
    """
    
    # simpleFuel.iterate_EPD("singleFuel",timestep)
    # multiFuel.iterate_EPD("multifuel",timestep)
    california.iterate_EPD("california",timestep)
    # california_wn.iterate_EPD("california_wn",timestep)
    # simpleFuel.iterate_LSTM("singleFuel",timestep)
    # multiFuel.iterate_LSTM("multiFuel",timestep)
    california.iterate_LSTM("california",timestep)
    # california_wn.iterate_LSTM("california_wn",timestep)
  
    
label_resized = tf.image.resize(np.expand_dims(label, axis=-1), (126, 126),"nearest").numpy()[:,:,0]

# simpleFuel.plotResults(timestep, label_resized, "Single Fuel", rootPath)
# multiFuel.plotResults(timestep, label_resized, "Multiple Fuel", rootPath)
california.plotResults(timestep, label_resized, "California", rootPath)
california.animateResults("LSTM", "vegetation")
# #california_wn.plotResults(timestep, label_resized, "California WN", rootPath)

    
print("=========Simulations Done===========")
