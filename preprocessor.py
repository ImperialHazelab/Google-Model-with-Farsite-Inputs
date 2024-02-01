"""Code to preprocess a single FARSITE case"""

import numpy as np
import matplotlib.pyplot as pl
from ChannelPrep import ChannelPrep
from ChannelPrep import CallGoogleModel

rootPath = "D:\OneDrive - Imperial College London\Documents\Coding Projects\FireScenarioGenerator\FireScenarioGenerator\Output_quad/"
moistureFiles = "fms"
burn_start = [2024,1,1,1300];       "Year, Month, Day, HHMM"
burn_duration = 10;                  "Hours"
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
prep.importFiles()

rasterSize=np.shape(prep.fuel)

EPD, LSTM = prep.normaliseAndStitchChannels("singleFuel", False)
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
    
pl.subplots(4,2)
    
for timestep in range(8,(burn_duration*steps_per_hour)):
    
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
    
    pl.subplot(2,4,1)
    pl.imshow(simpleFuel.channels_EPD[0,:,:,0])
    pl.title("Single, EPD")
    pl.subplot(2,4,5)
    pl.imshow(simpleFuel.channels_LSTM[0,7,:,:,0])
    pl.title("Single, LSTM")
    pl.subplot(2,4,2)
    pl.imshow(multiFuel.channels_EPD[0,:,:,0])
    pl.title("Multi, EPD")
    pl.subplot(2,4,6)
    pl.imshow(multiFuel.channels_LSTM[0,7,:,:,0])
    pl.title("Multi, LSTM")
    pl.subplot(2,4,3)
    pl.imshow(california.channels_EPD[0,:,:,0])
    pl.title("California, EPD")
    pl.subplot(2,4,7)
    pl.imshow(california.channels_LSTM[0,7,:,:,0])
    pl.title("California, LSTM")
    pl.subplot(2,4,4)
    pl.imshow(california_wn.channels_EPD[0,:,:,0])
    pl.title("California WN, EPD")
    pl.subplot(2,4,8)
    pl.imshow(california_wn.channels_LSTM[0,7,:,:,0])    
    pl.title("California WN, LSTM")
    
print("=========Simulations Done===========")
    


#tensor=tf.convert_to_tensor(np.array(channels_EPD),dtype=tf.float32)
#tf.saved_model.save(tensor, os.path.join(rootPath, "inputTensor.tsr"))



































    
    
