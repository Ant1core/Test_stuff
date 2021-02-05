### Test task 2###
# Script reads the json file with loan information(see description below), calculates and returns the total cumulative interest paid on a loan for the specified period.

#JSON file with loan data should look like this:
#{ "loan": number(int),  -- The amount of total loan
#  "rate": number(float), -- The annual interest rate
#  "years": number(int), -- years of debt
#  "pay_per_year": number(int), -- number of pays per year
#  "begin": number(int),  -- begin period (number in a row)
#  "end": number(int) -- end period (number in a row)
#}

import json

# open and read json
with open("d:/test_loan.json", "r") as read_file:
    data = json.load(read_file)

# reading dict
loan=int(data['loan'])
rate=float(data['rate'])
years=int(data['years'])
pay_per_year=int(data['pay_per_year'])
beg=int(data['begin'])
end=int(data['end'])

read_file.close()

# func to calculate the monthly payments
def count_month_pays(n_loan, n_rate, n_ppy, n_years):
    n_tp=n_years*n_ppy #total pays
    m_pay=n_loan*((n_rate/n_ppy)/(1-(1+(n_rate/n_ppy))**-n_tp))
    return m_pay

#func to calculate cumm. interest payed
def count_cum_int(n_loan, n_rate, n_years, n_ppy, beg, end):
    n_tp=n_years*n_ppy #total pays
    m_pay=count_month_pays(n_loan, n_rate, n_ppy, n_years) #monthly pay
    m_rate=n_ppy/n_rate # monthly rate
    
    cum_sum=(((n_loan-m_pay*m_rate)*(1+(1/m_rate))**(beg-1)+m_pay*m_rate)-((n_loan-m_pay*m_rate)*(1+(1/m_rate))**end+m_pay*m_rate))-m_pay*(end-beg+1)
    return round(cum_sum,2)

#count and print the target value
cs=count_cum_int(loan,rate,years,pay_per_year,beg,end)
print(cs)