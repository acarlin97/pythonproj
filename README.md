# HER Site Coastal Vulnerability Analysis (EGM722 Programming for GIS and Remote Sensing Project)
This project uses spatial analysis and Python programming to assess Historic Environment Records (HERs) at risk from coastal erosion and flooding in Northern Ireland. It identifies Sites and Monuments Record (SMR), Industrial Heritage Record (IHR), and Defence Heritage Record (DHR) locations within 150m and 15m of the coastline using buffer analysis, clipping and spatial joins.
## Repository
[Project Repository](https://github.com/acarlin97/pythonproj)
## Setup Instructions
1. Install the required software: [Git](https://git-scm.com/) and [Anaconda Naviagtor](https://www.anaconda.com/download).
2. Clone the repository: **git clone https://github.com/your_username/pythonproj.git**
3. Create the conda environment: **conda env create -f environment.yml**
4. Activate the environment: **conda activate pythonproj**
5. Run the main script in an IDE (e.g. PyCharm) or using the terminal or command prompt: **python proj_script.py**
## Main Dependencies
- geopandas
- cartopy
- matplotlib
- rasterio
- pyepsg

All dependencies will be installed automatically via the provided environment.yml file.
## Data
Place all required shapefiles and raster files into a folder e.g. ‘**data_files**’ in the root directory. Refer to the **PDF How-To Guide** for full data sourcing instructions and expected file names.
## Outputs
Running the script will generate:
- **Seven** figures: **Five** static maps of HER sites across NI and those within 150m/15m of the coastline and **two** grouped bar charts comparing HER site counts per county;
- **Three** CSV summary tables: Total HER counts and HER counts per county (150m and 15m)
- Shapefiles clipped to coastal buffers
## License
This project is licensed under the MIT License.