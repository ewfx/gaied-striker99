Instead place your source files here
1) Go to directory gaied-striker99\code\src\email-classification-api
2) pip install -r requirements.txt
3) go to scripts
4) run python main.py

Request curl to import to postman: 

1#

curl --location 'http://127.0.0.1:8000/classify-email/' \
--form 'file=@"/C:/Users/Admin/hackathon/email-classification-api/data/sample_emails/Student_Loan_Application_Request_-_Michael_T._Anderson.eml"'

2#

curl --location 'http://127.0.0.1:8000/train-model/' \
--header 'Content-Type: application/json' \
--data '{
"files":["Student_Loan_Application_Request_-_Emily_R._Johnson.eml"],
"labels":[{"type":"Loan_Application_and_Origination_Requests","subtype":"Student_Loan_Application"}]
}'

