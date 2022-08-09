# Handling missing data course project: Stack overflow survey trends
This project was completed for the codecademy course 'Handling missing 
data'. The scenario is as follows:

>You work for a staffing agency that specializes in finding qualified 
candidates for development roles. One of your latest clients is growing 
rapidly and wants to understand what kinds of developers they can hire, 
and to understand general trends of the technology market. Your 
organization has access to this Stack Overflow dataset, which consists of 
survey responses by developers all over the world for the last few years. 
Your project is to put together several statistical analyses about the 
community to educate your client about the potential hiring market for 
their company.

Skills employed in this project include:
- data exploration
- deletion
- multiple imputation

If you would like to view the accompanying jupyter notebook, please follow [this](https://nbviewer.org/github/radoya-panic/handling-missing-data-stack-overflow-survery-trends/blob/master/Stack%20overflow%20survey%20trends.ipynb) link to a notebook viewer page. 

# Summary
My analysis explored a number of patterns in the datasets including the geographical variation, educational background, and compensation level as it related to years having coded professionally. We have dealt with missing data values is different ways where appropriate. Some key findings include:

- after about 10 years of coding professionally, developer compensation is uncorrelated with experience
- assuming developers around the world are equally likely to use stackoverflow, then there are disproportionately many developers from the US overall, with Germany and the UK in approximately equal second place
- full stack and front/back-end developement is the most popular area of development
- there are a comparable amount of part-time and self-employed developers from Germany and the UK to the US, where the US greatly outnumbers them in the full-time employment scene
- the cost of back-end developers is independent of geography, but it isn't for most roles, such as for example for dta scientists, which are more expensive to hire in the US

Other trends that can be investigated, using the various visualisations produced, include:
- trends in developer langauge skill development
- trends in developer platform skill development
- trends in developer databse skill development
- the types of undegradute majors across developer types (using an interactive plot)
  - eg. data scientists are constituted by a lower proportion of compsci majors than other coding focused roles
- average compensation across different platforms, databases, and langauges that developers know

I also learned that the sweetviz package for EDA has some flaws. For example, there was a high correlation ratio for the Hobbyist and WorkWeekHrs features. However, after some digging, I discovered that it was because it replace nan's with zeros, which greatly skewed the accuracy of the calculation. I manually calculated the value, excluding nan values, and it was significantly smaller. This error was also present when calculating the association between categorical features via the uncertainty coefficient. I wrote and tested my own functions to compute the values and much more reasonable values were attained.
