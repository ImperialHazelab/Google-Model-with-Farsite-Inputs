"""Code to preprocess a single FARSITE case"""
import numpy as np
from ChannelPrep import ChannelPrep
from ChannelPrep import CallGoogleModel
from helpers import Farsite2Google
import tensorflow as tf
import matplotlib.pyplot as pl

rootPath = "D:\OneDrive - Imperial College London\Documents\Coding Projects\FireScenarioGenerator\FireScenarioGenerator/five_standard_cases\Output_singleFuel/"
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
label[label==0]=1
burnAreaPerTimestep = np.copy(label)/15
label[label>0]=0;
label[label<0]=1

rasterSize=np.shape(label)
resize_ratio = (126/rasterSize[0])
prep.simSteps = np.round(burn_duration * steps_per_hour * resize_ratio).astype(int)

prep.importFiles()
 
# prep.exportRawValues()

# EPD, LSTM = prep.normaliseAndStitchChannels("singleFuel", False)
# simpleFuel = CallGoogleModel(EPD, LSTM)

EPD, LSTM = prep.normaliseAndStitchChannels("multiFuel", False)
multiFuel = CallGoogleModel(EPD, LSTM)

EPD, LSTM = prep.normaliseAndStitchChannels("california", False)
california = CallGoogleModel(EPD, LSTM)

# EPD, LSTM = prep.normaliseAndStitchChannels("california_wn", False)
# california_wn = CallGoogleModel(EPD, LSTM)

#--------------------------RUN THE MODEL ITSELF------------------------------
    
label_resized = tf.image.resize(np.expand_dims(label, axis=-1), (126, 126),"nearest").numpy()[:,:,0]
label_per_timestep = np.zeros([np.shape(burnAreaPerTimestep)[0],np.shape(burnAreaPerTimestep)[1],prep.simSteps])
for i in range(1,prep.simSteps+1):
    label_per_timestep[np.logical_and(burnAreaPerTimestep<(i/resize_ratio),burnAreaPerTimestep>0),i-1] = 1

LSTM_per_timestep = np.zeros([prep.simSteps,np.shape(label_resized)[0],np.shape(label_resized)[1]])
label_per_timestep = tf.image.resize(label_per_timestep, (126, 126),"nearest").numpy()
for timestep in range(0,prep.simSteps-1):
    
    print("------ITERATION STEP ",timestep+1,"of ",prep.simSteps, " -----------")
    """
    Run all models concurrently
    
    """
    
    # simpleFuel.iterate_EPD("singleFuel",timestep)
    # multiFuel.iterate_EPD("multifuel",timestep)
    # california.iterate_EPD("california",timestep)
    # california_wn.iterate_EPD("california_wn",timestep)
    # simpleFuel.iterate_LSTM("singleFuel",timestep)
    multiFuel.iterate_LSTM("multiFuel",timestep)
    california.iterate_LSTM("california",timestep)
    
    LSTM_per_timestep[timestep]=np.copy(california.channels_LSTM[0,timestep+8,:,:,2])
    LSTM_per_timestep[timestep,LSTM_per_timestep[timestep]<0.1]=0
    
    # california_wn.iterate_LSTM("california_wn",timestep)
  
#--------------------------SETUP LABEL DATA--------------------------------
error = np.zeros(prep.simSteps)
error_image = np.zeros([prep.simSteps,126,126])
for i in range(0,prep.simSteps):
    error[i]=(np.count_nonzero(label_per_timestep[:,:,i])-np.count_nonzero(LSTM_per_timestep[i,:,:]))/((np.count_nonzero(label_per_timestep[:,:,i]))+1)
    error_image[i,:,:]=label_per_timestep[:,:,i]+LSTM_per_timestep[i,:,:]

fig, axs = pl.subplots(nrows=1, ncols=2, figsize=(8, 8), tight_layout=True)

detailedArrivalTime = california.resolveArrivalTime("LSTM")/resize_ratio
detailedArrivalTime_2 = detailedArrivalTime
detailedArrivalTime_2[detailedArrivalTime_2==0] = np.nan
detailedArrivalTime_2 = tf.image.resize(np.expand_dims(detailedArrivalTime_2, axis=-1), (np.shape(label)[0],np.shape(label)[1]),"gaussian").numpy()
ROS = CallGoogleModel.getRosFromArrivalTime(detailedArrivalTime_2,cellSize)

plot = axs[0].imshow(detailedArrivalTime_2,cmap="plasma")
axs[0].set_title(("ArrivalTime"))
pl.colorbar(plot)
axs[0].axis("on")

plot = axs[1].imshow(ROS,cmap="plasma")
axs[1].set_title("ROS")
pl.colorbar(plot)
axs[1].axis("on")
pl.show()

# simpleFuel.plotResults(timestep, label_resized, "Single Fuel", rootPath)
multiFuel.plotResults(timestep, label_resized, "Multiple Fuel", rootPath)
california.plotResults(timestep, label_resized, "California", rootPath)
california.animateResults("LSTM", "vegetation")
# #california_wn.plotResults(timestep, label_resized, "California WN", rootPath)

CallGoogleModel.examineDifference(50,label_per_timestep,LSTM_per_timestep)
CallGoogleModel.animateDifference(error_image)

fig = pl.plot(error*100)
pl.title(("LSTM model error"))
pl.xlabel("Simulated time (hours)")
#pl.xticks(np.arange(0, prep.simSteps+1, step=4), labels=[str((4*i*burn_duration/prep.simSteps).astype("int")) for i in range(0,((prep.simSteps+1)/4).astype("int")-1)])
pl.ylabel("Percent Error")
pl.show()

print("=========Simulations Done===========")



    
