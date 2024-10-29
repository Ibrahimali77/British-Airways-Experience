import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Reading data given
df = pd.read_csv('customer_booking.csv', encoding = 'ISO-8859-1')
print(df.head())

# Mapping flight days to a value

df['flight_day'].unique()

day_mapping = {
    'mon' : 1,
    'tue' : 2,
    'wed' : 3,
    'thu' : 4,
    'fri' : 5,
    'sat' : 6,
    'sun' : 7
}


# 3.1: Split the 'route' column into 'departure_airport' and 'arrival_airport'
df['departure_airport']  = df['route'].str[:3]
df['arrival_airport'] = df['route'].str[3:]

# Drop the original 'route' column
df.drop(columns=['route'], inplace=True)

#  simple label encoding.
flight_day_le = LabelEncoder()
df['flight_day_encoded'] = flight_day_le.fit_transform(df['flight_day'])

# Drop the original 'flight_day' column if it's not needed
df.drop(columns=['flight_day'], inplace=True)

trip_type_le = LabelEncoder()
df['trip_type_encoded'] = trip_type_le.fit_transform(df['trip_type'])
df.drop(columns =  ['trip_type'], inplace = True) 

sales_channel_le = LabelEncoder()
df['sales_channel_encoded'] = sales_channel_le.fit_transform(df['sales_channel'])
df.drop(columns= 'sales_channel', inplace = True)
# Label Encoding as an alternative to One-Hot Encoding
le_departure = LabelEncoder()
le_arrival = LabelEncoder()
df['departure_airport_encoded'] = le_departure.fit_transform(df['departure_airport'])
df['arrival_airport_encoded'] = le_arrival.fit_transform(df['arrival_airport'])
df.drop(columns = 'departure_airport', inplace = True)
df.drop(columns = 'arrival_airport', inplace= True)

booking_origin_le = LabelEncoder()
df['booking_origin_encoded'] = booking_origin_le.fit_transform(df['booking_origin'])
df.drop(columns= 'booking_origin', inplace = True)

# Scaling of Numerical Features
# --------------------------------------------
scaler = StandardScaler()

# Define numerical features to scale
numerical_features = ['purchase_lead', 'length_of_stay', 'flight_hour', 'flight_duration']

# Scale the numerical features
df[numerical_features] = scaler.fit_transform(df[numerical_features])


# Drop the columns that were encoded or not needed (you can adjust based on the encoding method)
# If using one-hot encoding, drop the label-encoded columns.
df = df.drop(columns=['departure_airport_encoded', 'arrival_airport_encoded'], errors='ignore')

# Select features (excluding 'booking_complete' which is the target)
X = df.drop(columns=['booking_complete'])
y = df['booking_complete']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Train a Model (Random Forest Example)
# ---------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Creating a new Dataframe based off of results

feature_importances = model.feature_importances_

features_df = pd.DataFrame({
    'Factor' : X.columns,
    'Importance' : feature_importances
})

features_df = features_df.sort_values(by= 'Importance', ascending = False)

# Step 7: Predict and Evaluate
# ----------------------------
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict = True)
print('Classification Report: ')
print(report)

report_df = pd.DataFrame(report).transpose()

# Displaying data
plt.figure(figsize=(10, 6))
sns.barplot(x= 'Importance', y='Factor', data=features_df, palette = 'viridis')
plt.title('Buying Prediction Results')
plt.xlabel('Importance Score')
plt.ylabel('Factors')
plt.show()

print("Preprocessed Data:")
print(df.head())
