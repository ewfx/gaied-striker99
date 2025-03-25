# Define your constants here
LABEL_MAP = {
    "Loan_Application_and_Origination_Requests": {
        "Student_Loan_Application": 0,        
    },
    "Loan_Insurance_and_Protection_Requests": {
        "Loan_Protection_Insurance_Request": 1,
        "Payment_Protection_Request": 2
    },
    "Loan_Reporting_and_Documentation_Requests": {
        "Loan_Statement_Request": 3,
        "Payment_Confirmation_Request": 4
    },
    "Loan_Account_Suspensions_and_Holds": {
        "Loan_Hold_Request": 5,
        "Suspension_of_Payment_Request": 6
    }
}

EXTRACTION_FIELDS = {
    0: ["Full Name", "Loan Amount Requested", "Date of Birth", "Social Security Number", "Permanent Address", "University Name", "Degree Program", "Expected Graduation Date"],
    1: ["Full Name", "Account Number", "Contact Information"],
    2: ["Full Name", "Account Number", "Contact Information"],
    3: ["Full Name", "Account Number", "Contact Information"],
    4: ["Account Number", "Transaction Date", "Payment Date", "Payment Amount"],
    5: ["Full Name", "Account Number", "Contact Information"],
    6: ["Full Name", "Account Number", "Contact Information"]
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