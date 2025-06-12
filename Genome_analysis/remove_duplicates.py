import pandas as pd

# Read the CSV file
df = pd.read_csv("/home/group_shyam01/Desktop/Autism_IBAB/datasets/CONCLUDED/concluded_2/dataset_dup_fill.csv")

# Remove leading/trailing whitespaces from all column names
df.columns = df.columns.str.strip()

# Print the DataFrame and columns
print(df.head())
print("Columns in dataset : " , df.columns)
print("Total no. of entries in dataset" , df.count())

# Find duplicates based on the 'Name' column
duplicates = df[df.duplicated(subset='Run', keep='first')]

# Print duplicates
print("List of the duplicates:\n" ,duplicates)
print("Total no. of duplicates " , duplicates.count())
#remove duplicates
df_unique = df.drop_duplicates(subset='Run', keep='first')
print("no. of entries after removing the duplicates : "  , df_unique.count())

#save this to csv
df_unique.to_csv("/home/group_shyam01/Desktop/Autism_IBAB/datasets/CONCLUDED/concluded_2/dataset_dup_fill_out.csv" , index=False)
print("file saved")




