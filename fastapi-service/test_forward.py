import requests

data = {
  "client": [
    {
      "client_id": 123,
      "age": 35,
      "gender": "F",
      "first_issue_date": "2022-01-10",
      "first_redeem_date": None
    }
  ],
  "purchases": [
    {
      "client_id": 123,
      "transaction_id": 1,
      "transaction_datetime": "2024-02-01 12:30:00",
      "purchase_sum": 540,
      "store_id": "54a4a11a29",
      "regular_points_received": 20,
      "express_points_received": 0,
      "regular_points_spent": 0,
      "express_points_spent": 0,
      "product_id": "9a80204f78",
      "product_quantity": 2,
      "trn_sum_from_iss": 540,
      "trn_sum_from_red": 0
    }
  ]
}


url = "http://localhost:8000/forward"

response = requests.post(url, json=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")