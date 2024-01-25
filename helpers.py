import numpy as np
import collections.abc
from typing import Dict, Tuple, List, Text
import os
from shapely import geometry
from shapely.prepared import prep
import fiona

class Farsite2Google:
    
    
    def __init__(self, rootPath, 
                 moistureFiles, 
                 burn_start, 
                 burn_duration, 
                 steps_per_hour, 
                 arraySize,
                 cellSize,
                 xllcorner,
                 yllcorner):
        self.rootPath = rootPath
        self.moistureFiles = moistureFiles
        self.burn_start = burn_start
        self.burn_duration = burn_duration
        self.steps_per_hour = steps_per_hour
        self.arraySize = arraySize
        self.cellSize = cellSize
        self.xllcorner = xllcorner
        self.yllcorner = yllcorner
    
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
        
        slope_North=slope*np.sin(np.radians(aspect))
        slope_East=slope*np.cos(np.radians(aspect))
                
        return slope_North, slope_East
    
    def expand_left_and_right_indeces(self, matrix):
        return np.expand_dims(np.expand_dims(matrix, axis=0), axis=-1)
    
    def expand_right_index(self, matrix):
        return np.expand_dims(matrix, axis=-1)
    
    
    def get_wind_N_S_from_wxs(self, wxs, datetime, duration, steps_per_hour):
        
        wind_North_full=np.ndarray(np.append(duration*steps_per_hour, self.arraySize));
        wind_East_full=np.ndarray(np.append(duration*steps_per_hour, self.arraySize));
        
        for i in range(duration):
            windMag, windDir = self._get_wind_profile_at_time(wxs, datetime)
            wind_N=windMag*np.cos(np.radians(windDir))
            wind_E=windMag*np.sin(np.radians(windDir))
            datetime[3]=datetime[3] + 100;                                   "Prep for next step"
            wind_North=np.full(self.arraySize,wind_N)
            wind_East=np.full(self.arraySize,wind_E)
            for j in range(4):
                wind_North_full[4*i+j,:,:] = wind_North
                wind_East_full[4*i+j,:,:] = wind_East
                
        return wind_North_full, wind_East_full
        
    def _get_wind_profile_at_time(self, wxs, datetime, skip_lines=4):
        file_path = os.path.join(self.rootPath, wxs)
        
        if not os.path.exists(file_path):
          raise FileNotFoundError(f'The file {file_path} does not exist.')
          return -1
    
        matrix = []
        try:
            print('Trying to load file: ', file_path)
            with open(file_path, 'r') as file:
                for _ in range(skip_lines): # Ignore headers
                    next(file)
                for line in file:
                    # Split the line into elements and convert them to floats
                    row = [float(element) for element in line.split()]
                    if row[0:4]==datetime:
                        matrix=row[6:8]
                        print(matrix)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"Error: An unexpected error occurred - {str(e)}")
        print('Done loading file: ', file_path)
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
    
    def reclassify_fuels_to_continuous(self, fuel):
       """Reclassify the 40 Scott/Burgan fuels to a semi-continuous range"""
       
       FUEL_LIST = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 121, 122, 123, 124,
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
          print('Trying to load file: ', file_path)
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
      print('Done loading file: ', file_path)
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
          
      moistures = np.ndarray(self.arraySize)
      moisturesMatrix=Farsite2Google.get_asc_file(self.rootPath, "moisture.fms",0)
      moisturesArray = moisturesMatrix[:, [0, text_to_integer_mapping.get(moistureType, -1)]]
    
      # Iterate through the replacement array
      for element, replacement_value in moisturesArray:
          # Find indices where the element matches in the fuels matrix
          indices = np.where(fuels == element)
    
          # Replace values in the matrix with the corresponding replacement value
          moistures[indices] = replacement_value
    
      return moistures.astype('float64')