# JP MORGAN QUANTITATIVE RESEARCH - ORIGINAL TASK DESCRIPTIONS

This document contains the original task descriptions from the JP Morgan Quantitative Research Virtual Experience Program.

---

## TASK 1: PRICE A COMMODITY STORAGE CONTRACT

### Background Information

You have been assigned to work on a new commodity trading product at JP Morgan to help price commodity storage contracts. This is an exploratory project aimed at establishing the feasibility of being able to use a predictive model to price monthly gas storage contracts.

The client can buy the same quantity of gas each month at two different dates and choose to either inject or withdraw gas from the storage facility on these dates. In doing so, they can lock in a profit on the difference between the gas price on these dates minus the costs associated with the storage facility and injection/withdrawal.

In your role as a quantitative researcher, you have been asked to prototype and test some functions to price these contracts in a Jupyter Notebook that can then be refined at a later date. These functions will be able to take in a date and return back the price of natural gas.

The desk will use these commodity contracts as an opportunity to make a margin on trading. Once a price has been decided upon the contract, the desk will trade based on that price.

### Here is your task

You will need to download the monthly natural gas price data. The monthly natural gas prices can be found in the Nat_Gas.csv file. You can use as many additional Python libraries and mathematical/statistical techniques as you wish.

You will need to create a prototype pricing model that can go through further validation and testing before being put into production. Eventually, this model may be the basis for fully automated quoting to clients, but for now, the desk will use it with manual oversight to explore options with the client.

Write a function that is able to use the data you created previously to price the contract. The client may want to choose multiple dates to inject or withdraw a set amount of gas, so your approach should generalize the explanation from before. Consider all the cash flows involved in the product.

You should write a function that takes these inputs and gives back the value of the contract. You can assume there is no transport delay and that interest rates are zero. Market holidays, weekends, and bank holidays need not be accounted for.

---

## TASK 2: PRICE A COMMODITY STORAGE CONTRACT (CONTINUED)

### Here is your task

You need to create a prototype pricing model that can go through further validation and testing before being put into production. Eventually, this model may be the basis for fully automated quoting to clients, but for now, the desk will use it with manual oversight to explore options with the client.

You should write a function that is able to use the data you created previously to price the contract. The client may want to choose multiple dates to inject and withdraw a set amount of gas, so your approach should generalize the explanation from before. Consider all the cash flows involved in the product.

The input parameters that should be taken into account for pricing are:

1. Injection dates
2. Withdrawal dates
3. The prices at which the commodity can be purchased/sold on those dates
4. The rate at which the gas can be injected/withdrawn
5. The maximum volume that can be stored
6. Storage costs

Write a function that takes these inputs and gives back the value of the contract. You can assume there is no transport delay and that interest rates are zero. Market holidays, weekends, and bank holidays need not be accounted for. Test your code by selecting a few sample inputs.

---

## TASK 3: LOAN DEFAULT PROBABILITY

### Background Information

You have now moved to a new team assisting the retail banking arm, which has been experiencing higher-than-expected default rates on personal loans. Loans are an important source of revenue for banks, but they are also associated with the risk that borrowers may default on their loans. A default occurs when a borrower stops making the required payments on a debt.

The risk team has begun to look at the existing book of loans to see if more defaults should be expected in the future and, if so, what the expected loss will be. They have collected data on customers and now want to build a predictive model that can estimate the probability of default based on customer characteristics. A better estimate of the number of customers defaulting on their loan obligations will allow us to set aside sufficient capital to absorb that loss. They have decided to work with you in the QR team to help predict the possible losses due to the loans that would potentially default in the next year.

Charlie, an associate in the risk team, who has been introducing you to the business area, sends you a small sample of their loan book and asks if you can try building a prototype predictive model, which she can then test and incorporate into their loss allowances.

### Here is your task

The risk manager has collected data on the loan borrowers. The data is in tabular format, with each row providing details of the borrower, including their income, total loans outstanding, and a few other metrics. There is also a column indicating if the borrower has previously defaulted on a loan. You must use this data to build a model that, given details for any loan described above, will predict the probability that the borrower will default (also known as PD: the probability of default). Use the provided data to train a function that will estimate the probability of default for a borrower. 

Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.

* You should produce a function that can take in the properties of a loan and output the expected loss.
* You can explore any technique ranging from a simple regression or a decision tree to something more advanced. You can also use multiple methods and provide a comparative analysis.

Submit your code below.

---

## TASK 4: CALCULATE EXPECTED LOSS

(This task was integrated with Task 3 - see above for the complete description including the Expected Loss calculation)

---

## TASK 5: FICO SCORE BUCKETING

### Background Information

Now that you are familiar with the portfolio and personal loans and risk are using your model as a guide to loss provisions for the upcoming year, the team now asks you to look at their mortgage book. They suspect that FICO scores will provide a good indication of how likely a customer is to default on their mortgage. Charlie wants to build a machine learning model that will predict the probability of default, but while you are discussing the methodology, she mentions that the architecture she is using requires categorical data. As FICO ratings can take integer values in a large range, they will need to be mapped into buckets. She asks if you can find the best way of doing this to allow her to analyze the data.

A FICO score is a standardized credit score created by the Fair Isaac Corporation (FICO) that quantifies the creditworthiness of a borrower to a value between 300 to 850, based on various factors. FICO scores are used in 90% of mortgage application decisions in the United States. The risk manager provides you with FICO scores for the borrowers in the bank's portfolio and wants you to construct a technique for predicting the PD (probability of default) for the borrowers using these scores.

### Here is your task

Charlie wants to make her model work for future data sets, so she needs a general approach to generating the buckets. Given a set number of buckets corresponding to the number of input labels for the model, she would like to find out the boundaries that best summarize the data. You need to create a rating map that maps the FICO score of the borrowers to a rating where a lower rating signifies a better credit score.

The process of doing this is known as quantization. You could consider many ways of solving the problem by optimizing different properties of the resulting buckets, such as the mean squared error or log-likelihood (see below for definitions). For background on quantization, see here.

**Mean squared error**

You can view this question as an approximation problem and try to map all the entries in a bucket to one value, minimizing the associated squared error. We are now looking to minimize the following:

MSE = Σ (value - bucket_mean)²

**Log-likelihood**

A more sophisticated possibility is to maximize the following log-likelihood function:

L = Σ [ ni × (ki/ni) × log(ki/ni) + ni × (1-ki/ni) × log(1-ki/ni) ]

Where bi is the bucket boundaries, ni is the number of records in each bucket, ki is the number of defaults in each bucket, and pi = ki / ni is the probability of default in the bucket. This function considers how rough the discretization is and the density of defaults in each bucket. This problem could be addressed by splitting it into subproblems, which can be solved incrementally (i.e., through a dynamic programming approach). For example, you can break the problem into two subproblems, creating five buckets for FICO scores ranging from 0 to 600 and five buckets for FICO scores ranging from 600 to 850.

**References:**
- Dynamic Programming: https://en.wikipedia.org/wiki/Dynamic_programming#Computer_programming
- Likelihood Function: https://en.wikipedia.org/wiki/Likelihood_function
- Quantization: https://en.wikipedia.org/wiki/Quantization_(signal_processing)

---

## DATA FILES PROVIDED

1. **Nat_Gas.csv** - Monthly natural gas price data (Tasks 1 & 2)
2. **Task_3_and_4_Loan_Data.csv** - Loan borrower data with default indicators (Tasks 3, 4, & 5)

---

## PROGRAM INFORMATION

**Program:** JP Morgan Quantitative Research Virtual Experience  
**Platform:** Forage  
**Link:** https://www.theforage.com/virtual-internships/prototype/TsaV48bxbfP5QrqsJ/quantitative-research

---

www.linkedin.com/in/rémi-berveiller

*These task descriptions are provided by JP Morgan Chase & Co. through the Forage platform for educational purposes.*
