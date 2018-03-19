def if_none_trivial(x):
    if x==0:
        return 0
    else:
        return 1

def violet_search_icecream_shop(stock, max_capacity,demands,n_days,overnight_fee,price,deliver_fee,total_expense=[],expense=0):
    delivery_min = max(0,demands[0]-stock)
    delivery_max = max(0,sum(demands) - stock)
    for delivery in range(delivery_min,delivery_max+1,1):
        expense_today = expense + if_none_trivial(delivery)*deliver_fee + delivery*price
        expense_today = expense_today + max(0,(stock+delivery-max_capacity))*overnight_fee
        stock_next = stock+delivery-demands[0]
        print("***********************")
        print("expense until yesterday: ",expense)
        print("expense until today: ", expense_today)
        print(n_days, "remains")
        if n_days>1:
            violet_search_icecream_shop(stock_next, max_capacity,demands[1:],n_days-1,overnight_fee,price,deliver_fee,total_expense,expense_today)
        else:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("total expense",expense_today)
            total_expense.append(expense_today)
            # yield(expense_today)

total_expense=[]
violet_search_icecream_shop(0,10,[1,2,1,4],4,1,3,4,total_expense=total_expense)
print(total_expense)
print("the optimum cost:", min(total_expense))


from collections import defaultdict

def code_preprocessing(delivery_code):
    code_dic = defaultdict(list)
    i = 0
    for code in delivery_code:
        crude = code.split('-',1)
        code_dic[crude[0]].append((crude[1],i))
        i = i+1
    print(code_dic)

code_dict = code_preprocessing(["123-2","2345-1","123-3","123-5","2345-5"])

def swarm_delivery(code_dict):
    bee = []
    for key,value in code_dict:
        bee.append(value)
    print(bee)

swarm_delivery(code_dict)
