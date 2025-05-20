from langchain_core.tools import tool
from typing import Optional, ClassVar
from langchain_core.tools import BaseTool
import traceback
import time
import concurrent.futures
from langchain_exa import ExaSearchRetriever, TextContentsOptions
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from .utils import Utility


class ResilientSearchTool(BaseTool):
    name: ClassVar[str] = "resilient_search"
    description: ClassVar[str] = (
        "Searches the internet using a fallback chain of DuckDuckGo, Tavily, Serper, and Exa."
    )

    duckduckgo: Optional[object] = None
    tavily: Optional[object] = None
    serp: Optional[object] = None
    exa: Optional[object] = None
    timeout: int = 5  # max per-tool wait time

    def __init__(self, duckduckgo=None, tavily=None, serp=None, exa=None, timeout=5):
        super().__init__()
        self.duckduckgo = duckduckgo
        self.tavily = tavily
        self.serp = serp
        self.exa = self.exa_search()
        self.timeout = timeout

    def exa_search(self):
        llm = Utility.llm()
        retriever = ExaSearchRetriever(
            k=3, text_contents_options=TextContentsOptions(max_characters=200)
        )
        prompt = PromptTemplate.from_template(
            """You are the trained assistant which see the content and elaborate using you own knowledge also about the query. This is the following context and the respective query:
            query: {query}
            <context>
            {context}
            </context"""
        )
        chain = (
            RunnableParallel({"context": retriever, "query": RunnablePassthrough()})
            | prompt
            | llm
        )
        return chain

    def _attempt_with_timeout(self, func, query: str) -> str:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, query)
            return future.result(timeout=self.timeout)

    def _run(self, query: str, run_manager=None) -> str:
        tools = []

        if self.duckduckgo:
            tools.append(("DuckDuckGo", lambda q: self.duckduckgo.invoke(q)))
        if self.serp:
            tools.append(("Serper", lambda q: self.serp.run(q)))
        if self.exa:
            tools.append(("Exa", lambda q: self.invoke(f"{q}?").content))
        if self.tavily:
            tools.append(
                (
                    "Tavily",
                    lambda q: self.tavily.invoke({"query": q})["results"][0]["content"],
                )
            )

        for name, func in tools:
            try:
                print(f"🔍 Trying {name}...")
                start = time.time()
                result = self._attempt_with_timeout(func, query)
                elapsed = round(time.time() - start, 2)
                print(f"✅ {name} succeeded in {elapsed}s.")
                return f"{result}"
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                print(traceback.format_exc())

        return "Failed to Access The Information, Please Try Again"


@tool
def get_schema_info() -> str:
    """Returns the schema of the Home Mortgage Disclosure Act dataset."""
    return """
    Available columns:
    - activity_year: categorical, The calendar year the data submission covers
    - lei: categorical, A financial institution's Legal Entity Identifier
    - derived_msa-md: categorical, The 5 digit derived MSA (metropolitan statistical area) or MD (metropolitan division) code. An MSA/MD is an area that has at least one urbanized area of 50,000 or more population.
    - state_code: categorical, Two-letter state code
    - county_code: categorical, State-county FIPS code
    - census_tract: categorical, 11 digit census tract number
    - derived_loan_product_type: categoricaL, Derived loan product type from Loan Type and Lien Status fields for easier querying of specific records
    - derived_dwelling_category: Categorical, Derived dwelling type from Construction Method and Total Units fields for easier querying of specific records
    - conforming_loan_limit: Categorical, Indicates whether the reported loan amount exceeds the GSE (government sponsored enterprise) conforming loan limit
    - derived_ethnicity: Categorical, Single aggregated ethnicity categorization derived from applicant/borrower and co-applicant/co-borrower ethnicity fields
    - derived_race: Categorical, Single aggregated race categorization derived from applicant/borrower and co-applicant/co-borrower race fields
    - derived_sex: Categorical, Single aggregated sex categorization derived from applicant/borrower and co-applicant/co-borrower sex fields
    - action_taken: Alphanumeric, The action taken on the covered loan or application
    - purchaser_type: Alphanumeric, Type of entity purchasing a covered loan from the institution
    - preapproval: Alphanumeric, Whether the covered loan or application involved a request for a preapproval of a home purchase loan under a preapproval program
    - loan_type: Alphanumeric, The type of covered loan or application
    - loan_purpose: Alphanumeric, The purpose of covered loan or application
    - lien_status: Alphanumeric, Lien status of the property securing the covered loan, or in the case of an application, proposed to secure the covered loan
    - reverse_mortgage: Alphanumeric, Whether the covered loan or application is for a reverse mortgage
    - open-end_line_of_credit: Alphanumeric, Whether the covered loan or application is for an open-end line of credit
    - business_or_commercial_purpose: Alphanumeric, Whether the covered loan or application is primarily for a business or commercial purpose
    - loan_amount: Alphanumeric, The amount of the covered loan, or the amount applied for
    - combined_loan_to_value_ratio: Alphanumeric, The ratio of the total amount of debt secured by the property to the value of the property relied on in making the credit decision
    - interest_rate: Alphanumeric, The interest rate for the covered loan or application
    - rate_spread: Alphanumeric, The difference between the covered loan's annual percentage rate (APR) and the average prime offer rate (APOR) for a comparable transaction as of the date the interest rate is set
    - hoepa_status: Alphanumeric, Whether the covered loan is a high-cost mortgage
    - total_loan_costs: Alphanumeric, The amount, in dollars, of total loan costs
    - total_points_and_fees: Alphanumeric, The total points and fees, in dollars, charged in connection with the covered loan
    - origination_charges: Alphanumeric, The total of all itemized amounts, in dollars, that are designated borrower-paid at or before closing
    - discount_points: Alphanumeric, The points paid, in dollars, to the creditor to reduce the interest rate
    - lender_credits: Alphanumeric, The amount, in dollars, of lender credits
    - loan_term: Alphanumeric, The number of months after which the legal obligation will mature or terminate, or would have matured or terminated
    - prepayment_penalty_term: Alphanumeric, The term, in months, of any prepayment penalty
    - intro_rate_period: Alphanumeric, The number of months, or proposed number of months in the case of an application, until the first date the interest rate may change after closing or account opening
    - negative_amortization: Alphanumeric, Whether the contractual terms include, or would have included, a term that would cause the covered loan to be a negative amortization loan
    - interest_only_payment: Alphanumeric, Whether the contractual terms include, or would have included, interest-only payments
    - balloon_payment: Alphanumeric, Whether the contractual terms include, or would have included, a balloon payment
    - other_nonamortizing_features: Alphanumeric, Whether the contractual terms include, or would have included, any term, other than those described in Paragraphs 1003.4(a)(27)(i) (ii), and (iii) that would allow for payments other than fully amortizing payments during the loan term
    - property_value: Alphanumeric, The value of the property securing the covered loan or, in the case of an application, proposed to secure the covered loan, relied on in making the credit decision
    - construction_method: Alphanumeric, Construction method for the dwelling
    - occupancy_type: Alphanumeric, Occupancy type for the dwelling
    - manufactured_home_secured_property_type: Alphanumeric, Whether the covered loan or application is, or would have been, secured by a manufactured home and land, or by a manufactured home and not land
    - manufactured_home_land_property_interest: Alphanumeric, The applicant’s or borrower’s land property interest in the land on which a manufactured home is, or will be, located
    - total_units: Alphanumeric, The number of individual dwelling units related to the property securing the covered loan or, in the case of an application, proposed to secure the covered loan
    - ageapplicant: Alphanumeric, The age of the applicant
    - multifamily_affordable_units: Alphanumeric, Reported values as a percentage, rounded to the nearest whole number, of the value reported for Total Units
    - income: Alphanumeric, The gross annual income, in thousands of dollars, relied on in making the credit decision, or if a credit decision was not made, the gross annual income relied on in processing the application
    - debt_to_income_ratio: Alphanumeric, The ratio, as a percentage, of the applicant’s or borrower’s total monthly debt to the total monthly income relied on in making the credit decision
    - applicant_credit_score_type: Alphanumeric, The name and version of the credit scoring model used to generate the credit score, or scores, relied on in making the credit decision
    - co-applicant_credit_score_type: Alphanumeric, The name and version of the credit scoring model used to generate the credit score, or scores, relied on in making the credit decision
    - applicant_ethnicity-1: Alphanumeric, Ethnicity of the applicant or borrower
    - applicant_ethnicity-2: Alphanumeric, Ethnicity of the applicant or borrower
    - applicant_ethnicity-3: Alphanumeric, Ethnicity of the applicant or borrower
    - applicant_ethnicity-4: Alphanumeric, Ethnicity of the applicant or borrower
    - applicant_ethnicity-5: Alphanumeric, Ethnicity of the applicant or borrower
    - co-applicant_ethnicity-1: Alphanumeric, Ethnicity of the first co-applicant or co-borrower
    - co-applicant_ethnicity-2: Alphanumeric, Ethnicity of the first co-applicant or co-borrower
    - co-applicant_ethnicity-3: Alphanumeric, Ethnicity of the first co-applicant or co-borrower
    - co-applicant_ethnicity-4: Alphanumeric, Ethnicity of the first co-applicant or co-borrower
    - co-applicant_ethnicity-5: Alphanumeric, Ethnicity of the first co-applicant or co-borrower
    - applicant_ethnicity_observed: Alphanumeric, Whether the ethnicity of the applicant or borrower was collected on the basis of visual observation or surname
    - co-applicant_ethnicity_observed: Alphanumeric, Whether the ethnicity of the first co-applicant or co-borrower was collected on the basis of visual observation or surname
    - applicant_race-1: Alphanumeric, Race of the applicant or borrower
    - applicant_race-2: Alphanumeric, Race of the applicant or borrower
    - applicant_race-3: Alphanumeric, Race of the applicant or borrower
    - applicant_race-4: Alphanumeric, Race of the applicant or borrower
    - applicant_race-5: Alphanumeric, Race of the applicant or borrower
    - co-applicant_race-1: Alphanumeric, Race of the first co-applicant or co-borrower
    - co-applicant_race-2: Alphanumeric, Race of the first co-applicant or co-borrower
    - co-applicant_race-3: Alphanumeric, Race of the first co-applicant or co-borrower
    - co-applicant_race-4: Alphanumeric, Race of the first co-applicant or co-borrower
    - co-applicant_race-5: Alphanumeric, Race of the first co-applicant or co-borrower
    - applicant_race_observed: Alphanumeric, Whether the race of the applicant or borrower was collected on the basis of visual observation or surname
    - co-applicant_race_observed: Alphanumeric, Whether the race of the first co-applicant or co-borrower was collected on the basis of visual observation or surname
    - applicant_sex: Alphanumeric, Sex of the applicant or borrower
    - co-applicant_sex: Alphanumeric, Sex of the first co-applicant or co-borrower
    - applicant_sex_observed: Alphanumeric, Whether the sex of the applicant or borrower was collected on the basis of visual observation or surname
    - co-applicant_sex_observed: Alphanumeric, Whether the sex of the first co-applicant or co-borrower was collected on the basis of visual observation or surname
    - applicant_age_above_62: Alphanumeric, Whether the applicant or borrower age is 62 or above
    - co-applicant_age: Alphanumeric, The age, in years, of the first co-applicant or co-borrower
    - co-applicant_age_above_62: Alphanumeric, Whether the co-applicant or co-borrower age is 62 or above
    - submission_of_application: Alphanumeric, Whether the applicant or borrower submitted the application directly to the financial institution
    - initially_payable_to_institution: Alphanumeric, Whether the obligation arising from the covered loan was, or, in the case of an application, would have been, initially payable to the financial institution
    - aus-1: Alphanumeric, The automated underwriting system(s) (AUS) used by the financial institution to evaluate the application
    - aus-2: Alphanumeric, The automated underwriting system(s) (AUS) used by the financial institution to evaluate the application
    - aus-3: Alphanumeric, The automated underwriting system(s) (AUS) used by the financial institution to evaluate the application
    - aus-4: Alphanumeric, The automated underwriting system(s) (AUS) used by the financial institution to evaluate the application
    - aus-5: Alphanumeric, The automated underwriting system(s) (AUS) used by the financial institution to evaluate the application
    - denial_reason-1: Alphanumeric, The principal reason, or reasons, for denial
    - denial_reason-2: Alphanumeric, The principal reason, or reasons, for denial
    - denial_reason-3: Alphanumeric, The principal reason, or reasons, for denial
    - denial_reason-4: Alphanumeric, The principal reason, or reasons, for denial
    - tract_population: Alphanumeric, Total population in tract
    - tract_minority_population_percent: Alphanumeric, Percentage of minority population to total population for tract, rounded to two decimal places
    - ffiec_msa_md_median_family_income: Alphanumeric, FFIEC Median family income in dollars for the MSA/MD in which the tract is located (adjusted annually by FFIEC)
    - tract_to_msa_income_percentage: Alphanumeric, Percentage of tract median family income compared to MSA/MD median family income
    - tract_owner_occupied_units: Alphanumeric, Number of dwellings, including individual condominiums, that are lived in by the owner
    - tract_one_to_four_family_homes: Alphanumeric, Dwellings that are built to houses with fewer than 5 families
    - tract_median_age_of_housing_units: Alphanumeric, Tract median age of homes
"""
