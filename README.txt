####################################
#     	     Description     	   #
####################################

This file loads in the dataset obtained from Kaggle. From there, it reads in the data and adds city, county, and state codes to each zip code. From there it imputes values for missing values through grouping.

After the data has been trimmed and imputed, the script enters the modeling section. This modeling section builds a predictive model for each Zip code.

The script returns a CSV of the completely clean filed along with a column of predicted values, as well as another CSV file of the forecasted values for the next 6 months.

Folder Structure:

- Main Folder
	- /data/
		Zip_time_series.csv
		county_to_zip.csv
		index.csv
	- data_loader.py
	- /final/

####################################
#      	    Installation	   #
####################################

Requirements:

- Python >= 3.7.0
- Pandas
- Numpy
- Scitkit-Learn
- Scipy
- Statsmodels
- Warnings
- Matplotlib
- OS

Dataset Requirements:

- https://www.kaggle.com/zillow/zecon (Zip_time_series.csv is the dataset being used)
- https://data.world/niccolley/us-zipcode-to-county-state/workspace/file?filename=ZIP-COUNTY-FIPS_2018-03.csv (rename to county_to_zip.csv)
- index.csv (this is provided in the package)

Place 'Zip_time_series.csv' and 'county_to_zip.csv' into the '/data/' folder (see structure below).

####################################
#      	     Execution             #
####################################

In a command prompt, navigate to the main folder.

From there, run the command "python data_loader.py" without the encasing quotes.

This will run the entire file with the file running print statements to update progress along the way.
