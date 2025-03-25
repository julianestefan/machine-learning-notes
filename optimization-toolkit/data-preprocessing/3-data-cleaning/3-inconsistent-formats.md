# Handling Inconsistent Data Formats

Inconsistent data formats are a common issue in real-world datasets that can significantly impact model performance if not properly addressed.

## Common Data Inconsistencies

1. **String format variations**: 'Yes', 'yes', 'Y', 'y' all representing the same concept
2. **Date and time formats**: Different standards (MM/DD/YYYY vs. DD/MM/YYYY)
3. **Numerical representations**: '1,000' vs '1000' vs '1.000'
4. **Units of measurement**: Mixed metric and imperial units
5. **Case sensitivity**: 'New York' vs 'new york'
6. **Special characters**: Extra spaces, punctuation, or non-printable characters
7. **Multiple languages**: 'Yes' vs 'Sí' vs 'Oui'

## String Cleaning and Standardization

### Basic String Cleaning

```python
import pandas as pd
import numpy as np
import re

# Load data
df = pd.read_csv('dataset.csv')

# Convert to lowercase
df['Category'] = df['Category'].str.lower()

# Strip whitespace
df['Description'] = df['Description'].str.strip()

# Replace special characters
df['Amount'] = df['Amount'].str.replace('$', '').str.replace(',', '')

# Convert to numeric
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Standardize common values
yes_no_mapping = {
    'yes': 'yes', 'y': 'yes', 'true': 'yes', '1': 'yes', 't': 'yes',
    'no': 'no', 'n': 'no', 'false': 'no', '0': 'no', 'f': 'no',
    np.nan: 'unknown'
}
df['HasInsurance'] = df['HasInsurance'].str.lower().map(yes_no_mapping)
```

### Regular Expressions for Complex Patterns

```python
# Extract numbers from text with mixed formats
def extract_number(text):
    if pd.isna(text):
        return np.nan
    
    # Try to find any number pattern
    match = re.search(r'(\d+[\.,]?\d*)', str(text))
    if match:
        # Replace comma with dot for consistent decimal separator
        return float(match.group(1).replace(',', '.'))
    return np.nan

# Apply to a column with mixed formats
df['Extracted_Value'] = df['Mixed_Format_Column'].apply(extract_number)

# Standardize phone numbers
def clean_phone(phone):
    if pd.isna(phone):
        return np.nan
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', str(phone))
    
    # Format consistently (example for US numbers)
    if len(digits_only) == 10:
        return f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
    elif len(digits_only) == 11 and digits_only[0] == '1':
        return f"({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
    else:
        return digits_only  # Return as is if doesn't match expected patterns

df['Phone_Standardized'] = df['Phone'].apply(clean_phone)
```

## Date and Time Standardization

```python
# Parse various date formats
def parse_date(date_str):
    if pd.isna(date_str):
        return np.nan
    
    # List of formats to try
    formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%d-%m-%Y',
        '%Y/%m/%d', '%b %d, %Y', '%d %b %Y', '%B %d, %Y'
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            pass
    
    # If all formats fail, try pandas default parser
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return np.nan

# Apply to a column with mixed date formats
df['Date_Standardized'] = df['Date'].apply(parse_date)

# Extract useful components
if 'Date_Standardized' in df.columns:
    df['Year'] = df['Date_Standardized'].dt.year
    df['Month'] = df['Date_Standardized'].dt.month
    df['Day'] = df['Date_Standardized'].dt.day
    df['Weekday'] = df['Date_Standardized'].dt.weekday
    
    # Create date features
    df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['Quarter'] = df['Date_Standardized'].dt.quarter
    
    # Calculate time differences
    df['Days_Since_Event'] = (pd.Timestamp.now() - df['Date_Standardized']).dt.days
```

## Numerical Data Standardization

```python
# Standardize mixed numeric formats
def standardize_numeric(value):
    if pd.isna(value):
        return np.nan
    
    # Convert to string to handle various types
    value_str = str(value)
    
    # Remove currency symbols, spaces, and other non-numeric characters
    # Keep decimal points and minus signs
    clean_value = re.sub(r'[^\d.-]', '', value_str)
    
    # Handle European format (replace decimal comma with point)
    if ',' in clean_value and '.' not in clean_value:
        clean_value = clean_value.replace(',', '.')
    elif ',' in clean_value and '.' in clean_value:
        # Format like 1,234.56
        clean_value = clean_value.replace(',', '')
    
    # Convert to float
    try:
        return float(clean_value)
    except:
        return np.nan

# Apply to columns with mixed numeric formats
numeric_cols = ['Price', 'Amount', 'Rate']
for col in numeric_cols:
    if col in df.columns:
        df[f'{col}_Standardized'] = df[col].apply(standardize_numeric)
```

## Unit Conversion

```python
# Define conversion factors
INCH_TO_CM = 2.54
LBS_TO_KG = 0.453592
MILE_TO_KM = 1.60934

# Function to convert height from various formats to cm
def standardize_height(height):
    if pd.isna(height):
        return np.nan
    
    height_str = str(height).lower()
    
    # Format: 5'10" or 5' 10"
    feet_inches_pattern = r"(\d+)\'[ ]?(\d+)[\"\']?"
    match = re.search(feet_inches_pattern, height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        total_inches = feet * 12 + inches
        return total_inches * INCH_TO_CM
    
    # Format: 5ft 10in or 5 ft 10 in
    feet_inches_text = r"(\d+)[ ]?ft[ ]?(\d+)[ ]?in"
    match = re.search(feet_inches_text, height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        total_inches = feet * 12 + inches
        return total_inches * INCH_TO_CM
    
    # Format: 180cm or 180 cm
    cm_pattern = r"(\d+)[ ]?cm"
    match = re.search(cm_pattern, height_str)
    if match:
        return float(match.group(1))
    
    # Assume it's just a number in cm
    try:
        return float(height_str)
    except:
        return np.nan

# Apply to height column
if 'Height' in df.columns:
    df['Height_cm'] = df['Height'].apply(standardize_height)

# Similarly for weight
def standardize_weight(weight):
    if pd.isna(weight):
        return np.nan
    
    weight_str = str(weight).lower()
    
    # Format: 150lbs or 150 lbs
    lbs_pattern = r"(\d+\.?\d*)[ ]?(?:lb|lbs)"
    match = re.search(lbs_pattern, weight_str)
    if match:
        return float(match.group(1)) * LBS_TO_KG
    
    # Format: 68kg or 68 kg
    kg_pattern = r"(\d+\.?\d*)[ ]?(?:kg|kgs)"
    match = re.search(kg_pattern, weight_str)
    if match:
        return float(match.group(1))
    
    # Assume it's just a number in kg
    try:
        return float(weight_str)
    except:
        return np.nan

# Apply to weight column
if 'Weight' in df.columns:
    df['Weight_kg'] = df['Weight'].apply(standardize_weight)
```

## Address and Location Standardization

```python
# Standardize state names/abbreviations
state_mapping = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}

# Invert the dictionary to also map full names to abbreviations
state_abbrev = {v: k for k, v in state_mapping.items()}

# Combine both for a complete mapping
state_complete = {**state_mapping, **state_abbrev}

# Function to standardize state names
def standardize_state(state):
    if pd.isna(state):
        return np.nan
    
    state_str = str(state).strip().title()
    
    # Check direct match
    if state_str in state_complete:
        return state_mapping.get(state_complete[state_str], state_str)
    
    # Handle partial matches
    for full_name, abbrev in state_abbrev.items():
        if full_name.startswith(state_str):
            return full_name
    
    return state_str  # Return as is if no match found

# Apply to state column
if 'State' in df.columns:
    df['State_Standardized'] = df['State'].apply(standardize_state)
```

## Handling Mixed Data Types

Sometimes columns contain mixed data types that need to be separated:

```python
# Example: Column contains both numeric and text separated by spaces
def split_mixed_column(value):
    if pd.isna(value):
        return np.nan, np.nan
    
    value_str = str(value)
    
    # Try to match a pattern like "123 units" or "123.45 kg"
    match = re.match(r'(\d+\.?\d*)\s*(.+)', value_str)
    if match:
        return float(match.group(1)), match.group(2).strip()
    
    # If not a mixed format, check if it's numeric
    try:
        return float(value_str), np.nan
    except:
        return np.nan, value_str

# Apply to a mixed column
if 'Mixed_Column' in df.columns:
    df['Numeric_Part'], df['Text_Part'] = zip(*df['Mixed_Column'].apply(split_mixed_column))
```

## Best Practices

1. **Document all transformations** applied to each column
2. **Create new columns** rather than overwriting original data
3. **Use consistent naming conventions** for standardized columns
4. **Validate standardized data** through visualization and summary statistics
5. **Consider automating standardization** for recurring data sources
6. **Apply transformations in a pipeline** for reproducibility
7. **Handle exceptions gracefully** with appropriate error messages
8. **Use domain knowledge** to validate and inform standardization 