# Define your constants here
LABEL_MAP = {
    "Loan_Application_and_Origination_Requests": {
        "Student_Loan_Application": 0,
        "Home_Loan_Application": 1
    },
    "Customer_Service_Inquiries": {
        "Account_Balance_Inquiry": 2,
        "Transaction_Dispute": 3
    }
}

EXTRACTION_FIELDS = {
    0: ["Full Name", "Loan Amount Requested", "Date of Birth", "Social Security Number", "Permanent Address", "University Name", "Degree Program", "Expected Graduation Date"],
    1: ["Full Name", "Loan Amount Requested", "Property Address", "Date of Birth", "Social Security Number"],
    2: ["Account Number", "Inquiry Date"],
    3: ["Account Number", "Transaction Date", "Transaction Amount", "Dispute Reason"]
}

def get_label_value(type_label, subtype_label):
    try:
        return LABEL_MAP[type_label][subtype_label]
    except KeyError:
        raise ValueError(f"Invalid type or subtype: {type_label}, {subtype_label}")
    
def get_nested_keys_for_value(d, value):
    for key, subdict in d.items():
        if isinstance(subdict, dict):
            for subkey, subvalue in subdict.items():
                if subvalue == value:
                    return key, subkey
    raise ValueError(f"Value {value} not found in the dictionary")