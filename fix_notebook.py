import json

# Read notebook
with open('notebooks/anomaly_detection_unsw.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Fix cell 6 (index 6, the preprocessing cell)
notebook['cells'][6]['source'] = [
    "# Split features and labels\n",
    "X_train = df_train.drop(columns=['attack_cat','label'])\n",
    "y_train = df_train['attack_cat']\n",
    "\n",
    "X_test = df_test.drop(columns=['attack_cat','label'])\n",
    "y_test = df_test['attack_cat']\n",
    "\n",
    "# Encode categorical features\n",
    "label_encoders = {}\n",
    "for col in X_train.select_dtypes(include='object').columns:\n",
    "    le = LabelEncoder()\n",
    "    X_train[col] = le.fit_transform(X_train[col])\n",
    "    label_encoders[col] = le\n",
    "    \n",
    "    # Handle unseen categories in test set\n",
    "    X_test[col] = X_test[col].map(lambda x: x if x in le.classes_ else le.classes_[0])\n",
    "    X_test[col] = le.transform(X_test[col])\n",
    "\n",
    "# Scale features\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
]

# Write back
with open('notebooks/anomaly_detection_unsw.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Fixed preprocessing cell")
