import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('kdd-file.csv')
df = df.sample(frac=0.1, random_state=42) # Use only 10% of the data

column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label"
]

df.columns = column_names

df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Normalize numerical columns in chunks (excluding the target column)
chunk_size = 20
columns = df.select_dtypes(include=['float64', 'int64']).columns

for i in range(0, len(columns), chunk_size):
	chunk = columns[i:i + chunk_size]
	df[chunk] = scaler.fit_transform(df[chunk])

X = df.drop('label', axis=1)
y = df['label']

# Split into training (80%) and testing (20%) datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training size: {X_train.shape}, Testing size: {X_test.shape}")

# Initialize and train model

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate on test data

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
