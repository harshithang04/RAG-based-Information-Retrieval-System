"""Additional ground-truth questions for "2022 Q3 AAPL.pdf" (Apple 10-Q, quarter ended June 25, 2022).

These 20 questions cover facts not in DEV_SET or HELDOUT_SET:
- Product segments (iPad, Wearables, Services)
- Geographic segments (Americas, Europe, Greater China, Japan, Rest of Asia Pacific)
- Gross margin percentages and narrative facts
- Cost of sales and tax provisions
"""

EXTRA_SET = [
    {
        "id": "ipad_sales_q3",
        "question": "What were iPad net sales in Q3 2022?",
        "evidence": ["iPad", "7,224"],
        "answer": "7,224",
    },
    {
        "id": "wearables_sales_q3",
        "question": "How much revenue did Wearables, Home and Accessories generate in the third quarter?",
        "evidence": ["Wearables, Home and Accessories", "8,084"],
        "answer": "8,084",
    },
    {
        "id": "services_revenue_q3",
        "question": "What were Services net sales for Q3 2022?",
        "evidence": ["Services", "19,604"],
        "answer": "19,604",
    },
    {
        "id": "americas_sales_q3",
        "question": "What were net sales in the Americas segment for Q3 2022?",
        "evidence": ["Americas", "37,472"],
        "answer": "37,472",
    },
    {
        "id": "europe_sales_q3",
        "question": "What were Europe segment net sales in the third quarter of 2022?",
        "evidence": ["Europe:", "19,287"],
        "answer": "19,287",
    },
    {
        "id": "greater_china_sales_q3",
        "question": "How much net sales did Apple report in Greater China during Q3 2022?",
        "evidence": ["Greater China:", "14,604"],
        "answer": "14,604",
    },
    {
        "id": "japan_sales_q3",
        "question": "What were Japan segment net sales in the third quarter of 2022?",
        "evidence": ["Japan:", "5,446"],
        "answer": "5,446",
    },
    {
        "id": "rest_asia_pacific_sales_q3",
        "question": "How much revenue did Apple earn in Rest of Asia Pacific in Q3 2022?",
        "evidence": ["Rest of Asia Pacific:", "6,150"],
        "answer": "6,150",
    },
    {
        "id": "americas_op_income_q3",
        "question": "What was the Americas segment operating income for Q3 2022?",
        "evidence": ["Americas:", "13,914"],
        "answer": "13,914",
    },
    {
        "id": "europe_op_income_q3",
        "question": "How much operating income did Europe segment generate in the third quarter?",
        "evidence": ["Europe:", "7,124"],
        "answer": "7,124",
    },
    {
        "id": "cost_of_sales_q3_extra",
        "question": "What was the cost of sales in Q3 2022?",
        "evidence": ["Cost of sales", "47,074"],
        "answer": "47,074",
    },
    {
        "id": "tax_provision_9m",
        "question": "What was the provision for income taxes for the nine months ended June 25, 2022?",
        "evidence": ["Provision for income taxes", "15,364"],
        "answer": "15,364",
    },
    {
        "id": "iphone_growth_driver",
        "question": "What drove the increase in iPhone net sales during Q3 2022?",
        "evidence": ["iPhone net sales increased during the third quarter", "new iPhone models"],
        "answer": "new iPhone models",
    },
    {
        "id": "mac_decline_reason",
        "question": "Why did Mac net sales decrease in the third quarter of 2022?",
        "evidence": ["Mac net sales decreased during the third quarter of 2022", "MacBook Air and iMac"],
        "answer": "lower net sales of MacBook Air and iMac",
    },
    {
        "id": "ipad_decline_q3",
        "question": "What was the primary reason for iPad net sales decline in Q3 2022?",
        "evidence": ["iPad net sales decreased during the third quarter of 2022", "iPad Pro"],
        "answer": "lower net sales of iPad Pro",
    },
    {
        "id": "products_margin_decline",
        "question": "Did Products gross margin increase or decrease during Q3 2022 compared to Q3 2021?",
        "evidence": ["Products gross margin decreased during the third quarter of 2022"],
        "answer": "decreased",
    },
    {
        "id": "total_gross_margin_q3",
        "question": "What was the total gross margin percentage in Q3 2022?",
        "evidence": ["Total gross margin percentage 43.3 %"],
        "answer": "43.3%",
    },
    {
        "id": "services_margin_percentage_q3",
        "question": "What was the gross margin percentage for Services in the third quarter of 2022?",
        "evidence": ["Services", "71.5 %"],
        "answer": "71.5%",
    },
    {
        "id": "products_margin_percentage_q3",
        "question": "What was the gross margin percentage for Products in Q3 2022?",
        "evidence": ["Products", "34.5 %"],
        "answer": "34.5%",
    },
    {
        "id": "nine_month_gross_margin",
        "question": "What was the total gross margin percentage for the nine-month period?",
        "evidence": ["Total gross margin percentage", "43.6 %"],
        "answer": "43.6%",
    },
]
