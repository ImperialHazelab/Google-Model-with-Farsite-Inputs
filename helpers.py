import numpy as np
from typing import Dict, Tuple, List, Text
import os
from shapely import geometry
from shapely.prepared import prep
import pandas as pd

class Farsite2Google:
    
    
    def __init__(self, rootPath, 
                 moistureFiles, 
                 burn_duration, 
                 steps_per_hour, 
                 arraySize,
                 cellSize,
                 xllcorner,
                 yllcorner,
                 simSteps):
        self.rootPath = rootPath
        self.moistureFiles = moistureFiles
        self.duration = burn_duration
        self.steps_per_hour = steps_per_hour
        self.arraySize = arraySize
        self.cellSize = cellSize
        self.xllcorner = xllcorner
        self.yllcorner = yllcorner
        self.simSteps = simSteps
    
    def burnMap(self, perimeter_poly):
        """Compute the fractional burn map for a given perimeter."""
        burn = np.zeros(self.arraySize)
        prepared_perimeter = prep(perimeter_poly)
    
        # Iterate over cells
        for i in range(self.arraySize[0]):
            for j in range(self.arraySize[1]):
                cell = geometry.box(
                     (j)   * self.cellSize + self.xllcorner,
                     (i)   * self.cellSize + self.yllcorner,
                     (j+1) * self.cellSize + self.xllcorner,
                     (i+1) * self.cellSize + self.yllcorner)
                if prepared_perimeter.contains(cell):
                    burn[self.yllcorner-i,j] = 1.0
                elif prepared_perimeter.intersects(cell):
                    burn[self.yllcorner-i,j] = cell.intersection(perimeter_poly).area / cell.area
    
        return burn
    
    def get_slope_N_S_from_wxs(self):
        
        slope=Farsite2Google.get_asc_file(self.rootPath, "slope.asc")
        aspect=Farsite2Google.get_asc_file(self.rootPath, "aspect.asc")
        
        slope_North=slope*np.cos(np.radians(aspect))
        slope_East=slope*np.sin(np.radians(aspect))
                
        return np.tan(np.radians(slope_North)), np.tan(np.radians(slope_East))
    
    def expand_left_and_right_indeces(matrix):
        return np.expand_dims(np.expand_dims(matrix, axis=0), axis=-1)
    
    def expand_right_index(matrix):
        return np.expand_dims(matrix, axis=-1)
    
    def expand_left_index(matrix):
        return np.expand_dims(matrix, axis=0)
    
    
    def get_wind_N_S_from_wxs(self, datetime, wxs):
                
        wind_North_full=np.ndarray(np.append(self.duration*self.steps_per_hour, self.arraySize));
        wind_East_full=np.ndarray(np.append(self.duration*self.steps_per_hour, self.arraySize));
                
        for i in range(self.duration):
            
            windMag, windDir = self._get_wind_profile_at_time(datetime, wxs)
            wind_N=windMag*np.cos(np.radians(180+windDir))
            wind_E=windMag*np.sin(np.radians(180+windDir))
            if datetime[3]==2300:
                datetime[3]=0
                datetime[2]=datetime[2]+1
            else:
                datetime[3]=datetime[3] + 100;                                   "Prep for next step"
            wind_North=np.full(self.arraySize,wind_N)
            wind_East=np.full(self.arraySize,wind_E)
            for j in range(self.steps_per_hour):
                wind_North_full[self.steps_per_hour*i+j,:,:] = wind_North
                wind_East_full[self.steps_per_hour*i+j,:,:] = wind_East
                
        # now we need to resize it to the scaled time
        wind_North_rescaled = np.ndarray(np.append(self.simSteps, self.arraySize))
        wind_East_rescaled = np.ndarray(np.append(self.simSteps, self.arraySize))
        time_ratio = self.simSteps / (self.duration*self.steps_per_hour)
        
        if (time_ratio < 1) :
            for i in range(self.simSteps):
                step = np.round(time_ratio*i).astype(int)
                wind_North_rescaled[i,:,:] = wind_North_full[step,:,:]
                wind_East_rescaled[i,:,:] = wind_East_full[step,:,:]
        elif (time_ratio >1):
            for i in range(self.simSteps):
                step = np.round(i/time_ratio).astype(int)
                wind_North_rescaled[i,:,:] = wind_North_full[step,:,:]
                wind_East_rescaled[i,:,:] = wind_East_full[step,:,:]
        else:
            wind_North_rescaled = wind_North_full
            wind_East_rescaled = wind_East_full
            
        return wind_North_rescaled, wind_East_rescaled
        
    def _get_wind_profile_at_time(self,datetime, wxs, skip_lines=4):
        file_path = os.path.join(self.rootPath, wxs)
                
        if not os.path.exists(file_path):
          raise FileNotFoundError(f'The file {file_path} does not exist.')
          return -1
        matrix = []
        try:
            print('Trying to load file: ', wxs)
            with open(file_path, 'r') as file:
                for _ in range(skip_lines): # Ignore headers
                    next(file)
                for line in file:
                    # Split the line into elements and convert them to floats
                    row = [float(element) for element in line.split()]
                    if row[0:4]==datetime:
                        matrix=row[7:9]
                        break
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"Error: An unexpected error occurred - {str(e)}")
        print('Done loading file: ', wxs)
        return matrix
    
    def _get_mean_std_from_matrix(matrix):
       """
       Calculate the mean and standard deviation of matrix data for normalisation
       """
        # Convert the matrix to a NumPy array
       matrix_array = np.array(matrix)
    
        # Calculate the mean and standard deviation
       mean_value = np.mean(matrix_array)
       std_deviation_value = np.std(matrix_array)
       return [mean_value, std_deviation_value]
    
    
    def norm_data_by_mean_var(self, data: any) -> any:
      """Normalizes data by removing the mean and dividing out the variance."""
      mean, std = Farsite2Google._get_mean_std_from_matrix(data)
      if mean == 0.0 and std == 0.0:
        return data
      if std == 0.0:
        return data - mean
      result = (data - mean) / std
      return result
  
    def norm_data_by_norms(self, data: any, model, channel) -> any:
        norms_singleFuel=[[0,0],
               [0,0],
               [0,0],
               [-1.16088,391.4318],
               [-0.16826,373.6882],
               [21.113,115.18],
               [20.665,122.3],
               [21.008,120.6939],
               [64.017,389.6247],
               [64.478,394.6595],
               [48.164,862.465],
               [260,18534.789],
               [116.164,9683.9357],
               [19.208,127.7187],
               [0.00453627,0.11082113],
               [0.00884734285,0.1045156],
               [0,0]]
        norms_multiFuel=[[0,0],
               [0,0],
               [0,0],
               [0.604685,421.4587],
               [-0.26755,407.305],
               [20.439,117.43],
               [20.17,121.57],
               [20.372,119.70],
               [63.21,410.557],
               [64.9629,403.8],
               [49.36,833.03],
               [263.76,18447.3],
               [114.947,9099.14],
               [19.521,131.22],
               [0.01387,0.108777],
               [0.008559,0.11821],
               [0,0]]
        norms_california=[[0,0],
               [0,0],
               [0,0],
               [0.301,417.234],
               [-0.0647,391.52],
               [20.669,117.062],
               [20.532,119.615],
               [20.568,121.68],
               [64.72,406.76],
               [64.033,414.37],
               [10.39,404.997],
               [47.31,9343.91],
               [6.125,335.35],
               [2.4873,32.097],
               [0.002422,0.010615],
               [-0.00044997,0.009587],
               [0,0]]
        norms_california_wn=[[0,0],
               [0,0],
               [0,0],
               [-1.11558,379.93],
               [0.3862,386.96],
               [20.320,121.64],
               [20.58,114.11],
               [20.52,118.33],
               [64.778,403.4],
               [64.882,411.64],
               [11.7,446.87],
               [53.32,10305.58],
               [6.7899,367.42],
               [2.7995,35.1733],
               [0.001234,0.0105837],
               [-0.0010052,0.009664],
               [0,0]]
        
        if model == "singleFuel":
            norms=norms_singleFuel
        elif model == "multiFuel":
            norms = norms_multiFuel
        elif model == "california":
            norms = norms_california
        elif model == "california_wn":
            norms = norms_california_wn
        
        mean = norms[channel-1][0]
        std = np.sqrt(norms[channel-1][1])
        if mean == 0.0 and std == 0.0:
          return data
        if std == 0.0:
          return data - mean
        result = (data - mean) / std
        return result
    
    def denorm_data_by_norms(data: any, model, channel) -> any:
        norms_singleFuel=[[0,0],
               [0,0],
               [0,0],
               [-1.16088,391.4318],
               [-0.16826,373.6882],
               [21.113,115.18],
               [20.665,122.3],
               [21.008,120.6939],
               [64.017,389.6247],
               [64.478,394.6595],
               [48.164,862.465],
               [260,18534.789],
               [116.164,9683.9357],
               [19.208,127.7187],
               [0.00453627,0.11082113],
               [0.00884734285,0.1045156],
               [0,0]]
        norms_multiFuel=[[0,0],
               [0,0],
               [0,0],
               [0.604685,421.4587],
               [-0.26755,407.305],
               [20.439,117.43],
               [20.17,121.57],
               [20.372,119.70],
               [63.21,410.557],
               [64.9629,403.8],
               [49.36,833.03],
               [263.76,18447.3],
               [114.947,9099.14],
               [19.521,131.22],
               [0.01387,0.108777],
               [0.008559,0.11821],
               [0,0]]
        norms_california=[[0,0],
               [0,0],
               [0,0],
               [0.301,417.234],
               [-0.0647,391.52],
               [20.669,117.062],
               [20.532,119.615],
               [20.568,121.68],
               [64.72,406.76],
               [64.033,414.37],
               [10.39,404.997],
               [47.31,9343.91],
               [6.125,335.35],
               [2.4873,32.097],
               [0.002422,0.010615],
               [-0.00044997,0.009587],
               [0,0]]
        norms_california_wn=[[0,0],
               [0,0],
               [0,0],
               [-1.11558,379.93],
               [0.3862,386.96],
               [20.320,121.64],
               [20.58,114.11],
               [20.52,118.33],
               [64.778,403.4],
               [64.882,411.64],
               [11.7,446.87],
               [53.32,10305.58],
               [6.7899,367.42],
               [2.7995,35.1733],
               [0.001234,0.0105837],
               [-0.0010052,0.009664],
               [0,0]]
        
        if model == "singleFuel":
            norms=norms_singleFuel
        elif model == "multiFuel":
            norms = norms_multiFuel
        elif model == "california":
            norms = norms_california
        elif model == "california_wn":
            norms = norms_california_wn
        
        mean = norms[channel][0]
        std = np.sqrt(norms[channel][1])
        if mean == 0.0 and std == 0.0:
          return data
        if std == 0.0:
          return data + mean
        result = data * std + mean
        return result
        
    
    def reclassify_fuels_to_continuous(self, fuel):
       """Reclassify the 40 Scott/Burgan fuels to a semi-continuous range"""
       
       FUEL_LIST = np.array([0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 121, 122, 123, 124,
                 141, 142, 143, 144, 145, 146, 147, 148, 149, 161, 162, 163, 164,
                 165, 181, 182, 183, 184, 185, 186, 187, 188, 189, 201, 202, 203,
                 204, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
       matrix = np.array(fuel)
       output = np.zeros(np.shape(fuel))
       i = 0
       for element in FUEL_LIST:
          # Find indices where the element matches in the fuels matrix
          indices = np.where(matrix == element)
          # Replace values in the matrix with the corresponding replacement value
          output[indices] = i
          i=i+1
       return output.astype('float64')
       
    
    def get_asc_file(rootPath, which_file: str, skip_lines=6):
      """
      Read a matrix from an ascii file.
    
      Parameters:
      - file_path (str): The path to the text file containing the matrix.
    
      Returns:
      - List[List[float]]: The matrix represented as a list of lists of floats.
      """
      file_path = os.path.join(rootPath, which_file)
      
      if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file {file_path} does not exist.')
        return -1
    
      matrix = []
      try:
          print('Trying to load file: ', which_file)
          with open(file_path, 'r') as file:
              for _ in range(skip_lines): # Ignore headers
                  next(file)
              for line in file:
                  # Split the line into elements and convert them to floats
                  row = [float(element) for element in line.split()]
                  matrix.append(row)
      except FileNotFoundError:
          print(f"Error: File '{file_path}' not found.")
      except Exception as e:
          print(f"Error: An unexpected error occurred - {str(e)}")
      print('Done loading file: ', which_file)
      return np.array(matrix).astype('float64')
    
    def get_moisture_raster(self, fuels, moistureType: Text):
      """Calculate the fuel moisture raster based on the fuel moisture file inputs
        
      Parameters:
      - fuels (numpy.ndarray): The original fuels matrix.
      - moisturesMatrix (numpy.ndarray): the input moisture file
      - moistureType: The moisture type matrix being created
      Returns:
      - numpy.ndarray: The matrix with replaced values.
      """
    
      text_to_integer_mapping = {
        "1hour": 1,
        "10hour": 2,
        "100hour": 3,
        "herbaceous": 4,
        "woody": 5
      }
          
      moistures = np.zeros(self.arraySize)
      moisturesMatrix=Farsite2Google.get_asc_file(self.rootPath, "moisture.fms",0)
      moisturesArray = moisturesMatrix[:, [0, text_to_integer_mapping.get(moistureType, -1)]]
      # Iterate through the replacement array
      for i in range(len(moisturesArray)):
          # Find indices where the element matches in the fuels matrix
          moistures[fuels == moisturesArray[i,0]]=moisturesArray[i,1]
          #indices = np.where(fuels == element)
          #print(indices)
          # Replace values in the matrix with the corresponding replacement value
          #moistures[indices] = replacement_value
      return moistures.astype('float64')
  
    
    def channels2excel(self, array, output_file):
        # Reshape the array to 17 separate HxW arrays
        print("Writing Channel data to Excel: ", output_file)
        H, W = np.shape(array[:,:,1])

        # Create a Pandas Excel writer
        with pd.ExcelWriter(self.rootPath + output_file) as writer:
            # Iterate over each part of the array
            for sheet_num in range(17):
                # Create a DataFrame from the flattened data
                df = pd.DataFrame(array[:,:,sheet_num])
    
                # Save the DataFrame to a separate sheet in the Excel file
                df.to_excel(writer, sheet_name=f'Sheet_{sheet_num}', index=False, header=False)
                
                
    